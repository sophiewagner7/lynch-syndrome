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
    strat_label = str(model.spec.strategy)
    cohort_label = f"{model.spec.cohort.gene}_{model.spec.cohort.sex}"
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
    if model.spec.strategy.is_NH or model.screening_log.empty:
        return pd.DataFrame(index=range(model.spec.cycles))

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
        index=range(model.spec.cycles),
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
        index=range(model.spec.cycles),
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
    return pd.Series(ly, index=range(model.spec.cycles), name="life_years")


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
    qalys_series = pd.Series(base_qalys, index=range(model.spec.cycles), name="qalys")

    # ----- colonoscopy disutility -----
    colo_specs = inputs.SCREENING_TEST_SPECS["colo"]
    if colo_specs["du_test"] > 0.0:
        # Screening colonoscopies this cycle
        if "colo" in model.screening_log.columns:
            screening_colo = (
                model.screening_log["colo"]
                .astype(float)
                .reindex(range(model.spec.cycles), fill_value=0.0)
            )
        else:
            screening_colo = pd.Series(0.0, index=range(model.spec.cycles))

        # Diagnostic colonoscopies for symptom-detected cancers
        if include_nonscreening_colo:
            diag_colo = pd.Series(
                model.symptom_detected.sum(axis=1), index=range(model.spec.cycles)
            )
        else:
            diag_colo = pd.Series(0.0, index=range(model.spec.cycles))

        n_colo_events = screening_colo.add(diag_colo, fill_value=0.0)

        # Subtract event-level QALY losses (already in QALYs per procedure)
        qalys_series = qalys_series - (n_colo_events * float(colo_specs["du_test"]))

    # ----- colo complication disutility ----
    if colo_specs["du_complication"] > 0.0:
        if "colo_complications" in model.complications_log:
            n_complications = (
                model.complications_log["colo_complications"]
                .astype(float)
                .reindex(range(model.spec.cycles), fill_value=0.0)
            )
        else:
            n_complications = pd.Series(0.0, index=range(model.spec.cycles))

        qalys_series = qalys_series - (
            n_complications * float(colo_specs["du_complication"])
        )

    return qalys_series


# ----- assemble qaly table for single run ----------
def build_costs_qalys_tables(
    model: MarkovModel,
    annual_discount_costs: Optional[float] = None,
    annual_discount_qalys: Optional[float] = None,
    use_half_cycle: bool = False,
    state_util_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_cycle_df, lifetime_df):
      per_cycle_df columns:
        - screening_<test> (one column per test) and screening_total
        - nonscreening_colo
        - colo_complications
        - total_cost
        - discount_costs
        - total_cost_disc
        - life years
        - qalys
        - discount_qalys
        - qalys_disc
      lifetime_df single-row summary with undiscounted & discounted totals.
    """
    n = model.spec.cycles

    # Costs
    screening_df = cost_screening_tests(model)
    if screening_df.empty:
        screening_df = pd.DataFrame(index=range(n))
        screening_df["total"] = 0.0
    screening_df = screening_df.add_prefix("screening_")

    nonscreening_colo = cost_nonscreening_colo(model)
    complications = cost_colo_complications(model)

    per_cycle = pd.DataFrame(index=range(n))
    per_cycle = per_cycle.join(screening_df, how="left")
    per_cycle["nonscreening_colo"] = nonscreening_colo
    per_cycle["colo_complications"] = complications

    # Totals (undiscounted)
    per_cycle["total_cost"] = per_cycle.filter(
        regex=r"^screening_|^nonscreening_colo$|^colo_complications$"
    ).sum(axis=1)

    # LY (per cycle)
    per_cycle["ly"] = life_years_per_cycle(model)

    # QALYs (per cycle)
    per_cycle["qalys"] = qalys_per_cycle(model, state_util_weights=state_util_weights)

    # Discount factors
    if annual_discount_costs is None:
        # try to pull from inputs, else default
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

    # Discounted
    per_cycle["total_cost_disc"] = per_cycle["total_cost"] * per_cycle["discount_costs"]
    per_cycle["qalys_disc"] = per_cycle["qalys"] * per_cycle["discount_qalys"]

    # Lifetime summaries
    lifetime = pd.DataFrame(
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
        }
    )

    # Optional: simple breakdowns
    lifetime["screening_cost"] = screening_df.filter(regex=r"^screening_").sum().sum()
    lifetime["nonscreening_colo_cost"] = per_cycle["nonscreening_colo"].sum()
    lifetime["colo_complication_cost"] = per_cycle["colo_complications"].sum()

    return per_cycle, lifetime


# ---------- IO ----------
# TODO: return model outputs of # cancers, # cancer screen-detected, stage dist, etc.
def save_run_outputs(
    model: MarkovModel,
    out_dir: Path,
    prefix: Optional[str] = None,
) -> Tuple[Path, Path]:
    """ """
    pass


def save_run_costs_utils(
    model: MarkovModel,
    per_cycle: pd.DataFrame,
    lifetime: pd.DataFrame,
    out_dir: Path,
    prefix: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Save per-cycle and lifetime CSVs. Returns the two paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label = prefix or _safe_strategy_label(model)
    p1 = out_dir / f"{label}__per_cycle.csv"
    p2 = out_dir / f"{label}__lifetime.csv"
    per_cycle.to_csv(p1, index_label="cycle")
    lifetime.to_csv(p2, index=False)
    return p1, p2
