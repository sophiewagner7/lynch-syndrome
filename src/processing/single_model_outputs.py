# costs.py
"""
Calculate costs and utilities for run.
"""
import re
from typing import Optional, Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from model.markov import MarkovModel
from configs import c
from configs import inputs

colo_specs = inputs.SCREENING_TEST_SPECS["colo"]


# ---------- helpers ----------
def _discount_factors(
    n_cycles: int,
    annual_rate: Optional[float],
    periods_per_year: int = 12,
    half_cycle_correction: bool = False,
) -> np.ndarray:
    """
    Returns length-n_cycles discount factors for an annual rate applied monthly.
    If half_cycle_correction=True, apply discount at mid-cycle (t - 0.5).
    """
    if annual_rate is None or annual_rate <= 0:
        return np.ones(n_cycles, dtype=float)
    t = np.arange(n_cycles, dtype=float)  # 0..n-1
    exponent = (t + (-0.5 if half_cycle_correction else 0.0)) / periods_per_year
    return 1.0 / np.power(1.0 + annual_rate, exponent)


def _safe_strategy_label(model: MarkovModel) -> str:
    """
    Create a short, file-safe label for a given run (strategy + cohort).
    e.g. MLH1_male_Colo_Q1_FIT_Q3_...
    """
    strat_label = str(model.strategy)
    cohort_label = f"{model.cohort.gene}_{model.cohort.sex}"
    label = f"{cohort_label}__{strat_label}".strip("_")
    label = re.sub(r"[^A-Za-z0-9_+.-]+", "_", label)
    return label


# ---------- cost calculators ----------
def cost_screening_tests(model: MarkovModel) -> pd.DataFrame:
    """
    Per-cycle cost for first-line tests (e.g., FIT, colo-first). Returns a DataFrame
    with one column per test in `model.screening_log`, plus a 'total' column.
    If NH, returns an empty DataFrame.
    """
    if model.strategy.is_NH or model.screening_log.empty:
        return pd.DataFrame(index=range(model.cycles))

    out = pd.DataFrame(index=model.screening_log.index)
    for test in model.screening_log.columns:
        test_cost = float(inputs.SCREENING_TEST_SPECS[test]["c_test"])
        tests_per_cycle = model.screening_log[test].astype(float)
        out[test] = test_cost * tests_per_cycle
    out["total"] = out.sum(axis=1)
    return out


def cost_nonscreening_colo(model: MarkovModel) -> pd.Series:
    """
    Per-cycle colonoscopy costs for *non-screening* colonoscopies applied to
    symptom-detected cancers. We treat "how many get colonoscopy" as the total
    symptom-detected inflow that cycle (sum across detected stages).
    """
    # symptom_detected is (cycles, 4). Sum across stages -> per-cycle count
    tests_per_cycle = model.symptom_detected.sum(axis=1)  # shape (cycles,)
    test_cost = inputs.SCREENING_TEST_SPECS["colo"]["c_test"]
    return pd.Series(
        tests_per_cycle * float(test_cost),
        index=range(model.cycles),
        name="nonscreening_colo",
    )


def cost_colo_complications(model: MarkovModel) -> pd.Series:
    """
    Per-cycle cost for colonoscopy complications. Uses `model.complications_log["colo_complications"]`.
    """
    complications_per_cycle = model.complications_log["colo_complications"]
    complication_cost = inputs.SCREENING_TEST_SPECS["colo"]["c_complication"]

    return pd.Series(
        complications_per_cycle.astype(float) * float(complication_cost),
        index=range(model.cycles),
        name="colo_complications",
    )


# ---------- life years -----------------
def life_years_per_cycle(model: MarkovModel) -> pd.Series:
    """
    Simple life years per cycle (no utilities, no discounting).
    LY_t = (alive_pop_t / POPULATION_SIZE) * cycle_length_years
    """
    alive_pop = model.D[:, c.alive_states_idx].sum(axis=1)  # shape (cycles,)
    ly = alive_pop * c.CYCLE_LENGTH
    return pd.Series(ly, index=range(model.cycles), name="life_years")


def total_life_years(model: MarkovModel) -> float:
    """
    Lifetime (undiscounted) life years = sum over cycles.
    """
    return float(life_years_per_cycle(model).sum())


# ---------- utilities / QALYs ----------
def qalys_per_cycle(
    model: MarkovModel,
    state_util_weights: dict[str, float] | None = None,
    include_nonscreening_colo: bool = True,
) -> pd.Series:
    """
    QALYs per cycle with an optional per-colonoscopy disutility.
    Total QALYs (scale of population)

    Base QALYs:
        QALY_t = (sum_s D[t, s] * u_s / POP_SIZE) * cycle_length_years

    Colonoscopy disutility:
        If colo_disutil_qaly_per_event > 0, subtract:
            (n_colonoscopy_events_t) * colo_disutil_qaly_per_event
        from QALY_t. The disutility parameter should already be in QALYs per event
        (i.e., utility decrement * duration in years for the event).

    Counts of colonoscopy events per cycle:
        - Screening colonoscopies: model.screening_log['colo'] if present (0 otherwise)
        - Diagnostic (non-screening) colonoscopies: sum over model.symptom_detected[t, :]
          (enabled when include_nonscreening_colo=True).
    """
    # ----- base utilities -----
    D = model.D
    n_states = c.n_states

    if state_util_weights:
        u = np.zeros(n_states, dtype=float)
        for name, w in state_util_weights.items():
            u[c.health_states_stoi[name]] = float(w)
    else:
        # Default: alive non-cancer (screening states) utility = 1.0
        u = np.zeros(n_states, dtype=float)
        u[c.screening_states_idx] = 1.0
        u[c.health_states_stoi["d_stage_1"]] = 0.9
        u[c.health_states_stoi["d_stage_2"]] = 0.7
        u[c.health_states_stoi["d_stage_3"]] = 0.3
        u[c.health_states_stoi["d_stage_4"]] = 0.1

    base_qalys = (D @ u) * c.CYCLE_LENGTH
    qalys_series = pd.Series(base_qalys, index=range(model.cycles), name="qalys")

    # ----- colonoscopy disutility -----
    colo_specs = inputs.SCREENING_TEST_SPECS["colo"]
    if colo_specs["du_test"] > 0.0:
        # Screening colonoscopies this cycle
        if "colo" in model.screening_log.columns:
            screening_colo = (
                model.screening_log["colo"]
                .astype(float)
                .reindex(range(model.cycles), fill_value=0.0)
            )
        else:
            screening_colo = pd.Series(0.0, index=range(model.cycles))

        # Diagnostic colonoscopies for symptom-detected cancers
        if include_nonscreening_colo:
            diag_colo = pd.Series(
                model.symptom_detected.sum(axis=1), index=range(model.cycles)
            )
        else:
            diag_colo = pd.Series(0.0, index=range(model.cycles))

        n_colo_events = screening_colo.add(diag_colo, fill_value=0.0)

        # Subtract event-level QALY losses (already in QALYs per procedure)
        qalys_series = qalys_series - (n_colo_events * float(colo_specs["du_test"]))

    # ----- colo complication disutility ----
    if colo_specs["du_complication"] > 0.0:
        if "colo_complications" in model.complications_log:
            n_complications = (
                model.complications_log["colo_complications"]
                .astype(float)
                .reindex(range(model.cycles), fill_value=0.0)
            )
        else:
            n_complications = pd.Series(0.0, index=range(model.cycles))

        qalys_series = qalys_series - (
            n_complications * float(colo_specs["du_complication"])
        )

    return qalys_series


# ----- assemble qaly table for single run ----------
def build_cancer_outcomes(model: MarkovModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_cycle_df, lifetime_df) with cancer counts by detection mode and stage,
    totals, and deaths. Counts are population-level (not per-person).
    """
    n = model.cycles
    per_cycle = pd.DataFrame(index=range(n))
    stage_labels = ["stage1", "stage2", "stage3", "stage4"]

    # --- screen-detected by stage ---
    if hasattr(model, "screen_detected") and model.screen_detected is not None:
        for i, s in enumerate(stage_labels):
            per_cycle[f"crc_screen_{s}"] = model.screen_detected[:, i].astype(float)
        per_cycle["crc_incident_screen"] = per_cycle[
            [f"crc_screen_{s}" for s in stage_labels]
        ].sum(axis=1)
    else:
        for s in stage_labels:
            per_cycle[f"crc_screen_{s}"] = 0.0
        per_cycle["crc_incident_screen"] = 0.0

    # --- symptom-detected by stage ---
    if hasattr(model, "symptom_detected") and model.symptom_detected is not None:
        for i, s in enumerate(stage_labels):
            per_cycle[f"crc_symptom_{s}"] = model.symptom_detected[:, i].astype(float)
        per_cycle["crc_incident_symptom"] = per_cycle[
            [f"crc_symptom_{s}" for s in stage_labels]
        ].sum(axis=1)
    else:
        for s in stage_labels:
            per_cycle[f"crc_symptom_{s}"] = 0.0
        per_cycle["crc_incident_symptom"] = 0.0

    # --- totals & deaths ---
    per_cycle["crc_incident_total"] = (
        per_cycle["crc_incident_screen"] + per_cycle["crc_incident_symptom"]
    )

    # CRC deaths per cycle (fallback to 0 if index not present)
    death_col = 13
    try:
        per_cycle["crc_deaths"] = model.I[:, death_col].astype(float)
    except Exception:
        per_cycle["crc_deaths"] = 0.0

    # ---------- lifetime (single row) ----------
    lifetime = pd.DataFrame(
        {
            "crc_incidents_total": [per_cycle["crc_incident_total"].sum()],
            "crc_incidents_screen": [per_cycle["crc_incident_screen"].sum()],
            "crc_incidents_symptom": [per_cycle["crc_incident_symptom"].sum()],
            "crc_deaths": [per_cycle["crc_deaths"].sum()],
        }
    )

    # stage totals & percents by detection mode
    for src, prefix in [("screen", "crc_screen_"), ("symptom", "crc_symptom_")]:
        denom = float(lifetime[f"crc_incidents_{src}"].iloc[0])
        for i, s in enumerate(stage_labels, start=1):
            count = float(per_cycle[f"{prefix}{s}"].sum())
            lifetime[f"{src}_stage_{i}_count"] = [count]
            lifetime[f"{src}_stage_{i}_pct"] = [count / denom if denom > 0 else 0.0]

    # --- stage totals & percents across BOTH detection modes ---
    denom_total = float(lifetime["crc_incidents_total"].iloc[0])
    for i, s in enumerate(stage_labels, start=1):
        count_total = float(per_cycle[f"crc_screen_{s}"].sum()) + float(
            per_cycle[f"crc_symptom_{s}"].sum()
        )
        lifetime[f"total_stage_{i}_count"] = [count_total]
        lifetime[f"total_stage_{i}_pct"] = [
            count_total / denom_total if denom_total > 0 else 0.0
        ]

    # per-person versions (simple, readable)
    pop = float(c.POPULATION_SIZE)
    lifetime["per_person_crc_incidents_total"] = lifetime["crc_incidents_total"] / pop
    lifetime["per_person_crc_incidents_screen"] = lifetime["crc_incidents_screen"] / pop
    lifetime["per_person_crc_incidents_symptom"] = (
        lifetime["crc_incidents_symptom"] / pop
    )
    lifetime["per_person_crc_deaths"] = lifetime["crc_deaths"] / pop

    # screening counts
    n_colos = 0.0
    n_alt = 0.0
    if hasattr(model, "screening_log") and not model.screening_log.empty:
        if "colo" in model.screening_log.columns:
            n_colos = float(model.screening_log["colo"].sum())
        for test in getattr(c, "ALT_TESTS", []):
            if test in model.screening_log.columns:
                n_alt += float(model.screening_log[test].sum())
    lifetime["screening_colonoscopies"] = n_colos
    lifetime["screening_alt_tests"] = n_alt
    lifetime["per_person_screening_colonoscopies"] = n_colos / pop
    lifetime["per_person_screening_alt_tests"] = n_alt / pop

    return per_cycle, lifetime


def build_costs_qalys_tables(
    model: MarkovModel,
    annual_discount_costs: Optional[float] = None,
    annual_discount_qalys: Optional[float] = None,
    use_half_cycle: bool = False,
    state_util_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_cycle_df, lifetime_df) that include:
      - cancer outcomes (incidents by detection/stage, totals, deaths)
      - screening, non-screening colo, complication costs
      - total costs (undiscounted + discounted)
      - LY, QALYs (undiscounted + discounted)
      - lifetime rollups incl. per-person values
    """
    n = model.cycles

    # ----- outcomes first -----
    outcomes_per_cycle, outcomes_lifetime = build_cancer_outcomes(model)

    # ----- costs -----
    screening_df = cost_screening_tests(model)
    if screening_df.empty:
        screening_df = pd.DataFrame(index=range(n))
        screening_df["total"] = 0.0
    screening_df = screening_df.add_prefix(
        "screening_"
    )  # ex: screening_colo, screening_total

    nonscreening_colo = cost_nonscreening_colo(model)
    complications = cost_colo_complications(model)

    # ----- assemble per-cycle -----
    per_cycle = outcomes_per_cycle.join(screening_df, how="left")
    per_cycle["nonscreening_colo"] = nonscreening_colo
    per_cycle["colo_complications"] = complications

    # totals (undiscounted costs)
    per_cycle["total_cost"] = per_cycle.filter(
        regex=r"^screening_|^nonscreening_colo$|^colo_complications$"
    ).sum(axis=1)

    # life-years and QALYs
    per_cycle["ly"] = life_years_per_cycle(model)
    per_cycle["qalys"] = qalys_per_cycle(model, state_util_weights=state_util_weights)

    # discount factors
    if annual_discount_costs is None:
        annual_discount_costs = getattr(
            inputs, "DISCOUNT_RATE_COSTS", getattr(inputs, "DISCOUNT_RATE", 0.03)
        )
    if annual_discount_qalys is None:
        annual_discount_qalys = getattr(
            inputs, "DISCOUNT_RATE_QALY", getattr(inputs, "DISCOUNT_RATE", 0.03)
        )

    per_cycle["discount_costs"] = _discount_factors(
        n, annual_discount_costs, half_cycle_correction=use_half_cycle
    )
    per_cycle["discount_qalys"] = _discount_factors(
        n, annual_discount_qalys, half_cycle_correction=use_half_cycle
    )

    # discounted totals
    per_cycle["total_cost_disc"] = per_cycle["total_cost"] * per_cycle["discount_costs"]
    per_cycle["qalys_disc"] = per_cycle["qalys"] * per_cycle["discount_qalys"]

    # ----- lifetime rollup -----
    lifetime_costs = pd.DataFrame(
        {
            "total_cost": [per_cycle["total_cost"].sum()],
            "total_cost_disc": [per_cycle["total_cost_disc"].sum()],
            "total_qalys": [per_cycle["qalys"].sum()],
            "total_qalys_disc": [per_cycle["qalys_disc"].sum()],
            "total_ly": [per_cycle["ly"].sum()],
            "per_person_cost": [
                per_cycle["total_cost"].sum() / float(c.POPULATION_SIZE)
            ],
            "per_person_cost_disc": [
                per_cycle["total_cost_disc"].sum() / float(c.POPULATION_SIZE)
            ],
            "per_person_qalys": [per_cycle["qalys"].sum() / float(c.POPULATION_SIZE)],
            "per_person_qalys_disc": [
                per_cycle["qalys_disc"].sum() / float(c.POPULATION_SIZE)
            ],
            "per_person_ly": [per_cycle["ly"].sum() / float(c.POPULATION_SIZE)],
            "screening_cost": [screening_df.filter(regex=r"^screening_").sum().sum()],
            "nonscreening_colo_cost": [per_cycle["nonscreening_colo"].sum()],
            "colo_complication_cost": [per_cycle["colo_complications"].sum()],
        }
    )

    # merge outcomes + costs into a single lifetime row
    lifetime = pd.concat(
        [outcomes_lifetime.reset_index(drop=True), lifetime_costs], axis=1
    )

    return per_cycle, lifetime


def summarize_cancers(model: MarkovModel) -> float:
    """
    Quick total CRC count (sum of inflows to u/d-stage columns if your indices 7:11
    are cancer states; adjust as needed).
    """
    return float(model.I[:, 7:11].sum())


def add_crc_vs_nh(
    df: pd.DataFrame, base_strategy: str = "NH", total_col: str = "crc_incidents_total"
) -> pd.DataFrame:
    """
    Adds:
      - crc_vs_NH_abs: absolute reduction in total cancers vs NH
      - crc_vs_NH_pct: percent reduction vs NH
    Expects columns: ['strategy', total_col]
    """
    nh = df.loc[df["strategy"] == base_strategy, total_col]
    if nh.empty:
        df["crc_vs_NH_abs"] = np.nan
        df["crc_vs_NH_pct"] = np.nan
        return df

    nh_val = float(nh.iloc[0])
    df["crc_vs_NH_abs"] = nh_val - df[total_col]
    df["crc_vs_NH_pct"] = (df["crc_vs_NH_abs"] / nh_val) * 100.0
    return df


# ---------- IO ----------
def save_run_arrays(
    model: MarkovModel,
    out_dir: Path,
    prefix: Optional[str] = None,
) -> Tuple[Path, Path, Path, Path]:
    """
    Save matrices D, I, P, R.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label = prefix or _safe_strategy_label(model)
    p1 = out_dir / f"{label}__D.csv"
    p2 = out_dir / f"{label}__I.csv"
    p3 = out_dir / f"{label}__P.csv"
    p4 = out_dir / f"{label}__R.csv"
    pd.DataFrame(model.D).to_csv(p1)
    pd.DataFrame(model.I).to_csv(p2)
    pd.DataFrame(model.P).to_csv(p3)
    pd.DataFrame(model.R).to_csv(p4)

    return p1, p2, p3, p4


def save_model_outputs(
    model: MarkovModel,
    df: pd.DataFrame,
    percycle: bool,
    out_dir: Path,
    suffix: str,
    prefix: Optional[str] = None,
) -> Path:
    """
    Save per-cycle and lifetime CSVs. Returns the two paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label = prefix or _safe_strategy_label(model)
    p = out_dir / f"{label}__{suffix}.csv"
    if percycle:
        df.to_csv(p, index_label="cycle")
    else:
        df.to_csv(p, index=False)
    return p


SUMMARY_COLS = [
    "gene",
    "sex",
    "strategy",
    "crc_incidents_total",
    "crc_deaths",
    "crc_vs_NH_pct",
    "per_person_screening_colos",
    "per_person_screening_alt_tests",
    "total_ly",
    "per_person_ly",
    "per_person_qalys_disc",
]


def save_lifetime_summary(
    lifetime_all: pd.DataFrame, out_dir: Path, base_name: str, sum_cols=SUMMARY_COLS
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Full file (everything)
    lifetime_all.to_csv(out_dir / f"{base_name}__lifetime_full.csv", index=False)

    # 2) Skinny summary (pick the cols that exist)
    summary_cols = [c for c in sum_cols if c in lifetime_all.columns]
    summary_cols += [
        c
        for c in lifetime_all.columns
        if c.startswith("total_stage_") and c.endswith("_pct")
    ]
    outcome_cols = [c for c in summary_cols if c in lifetime_all.columns]

    lifetime_all[outcome_cols].to_csv(
        out_dir / f"{base_name}__lifetime_summary.csv", index=False
    )
