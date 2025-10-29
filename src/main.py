# main.py
"""
Run Markov model(s) based on strategy and cohort inputs.

Run modes:
- "by_gene_strategies": for each (gene, sex) run every strategy; compute ICERs within each cohort.
- "by_strategy_all_genes": pick one strategy and run it across multiple genes/sexes; no ICERs.

To use this, cd into src and run with:
python3 main.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from configs import *
from model import cohort, markov, cancer_microsim
from processing.single_model_outputs import (
    build_costs_qalys_tables,
    save_model_outputs,
    save_run_arrays,
    add_crc_vs_nh,
    save_lifetime_summary,
)
from processing.icer import compute_icers, summarize_for_icer

# ------------------ USER CONFIG ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "20251006_1000"

STRATEGIES_FILE = "pms2_strats.json"
TMAT_STACK_FILE = "PMS2_20241006_1100.npy"

# Select run mode:
RUN_MODE = "single_gene_all_strategies"  # OPTIONS: "single_gene_all_strategies", "single_strategy_all_genes"

# Cohort selections:
GENES_TO_RUN = ["PMS2"]  # options from c.GENES, e.g. ["MLH1","MSH2","MSH6","PMS2"]
SEXES_TO_RUN = ["male", "female"]  # ["male","female"]
c.GENES = ["PMS2"]

# Save options:
SAVE_LIFETIME_IND = False  # set True to save each lifetime output separately
SAVE_PER_CYCLE = False  # set True to write per-cycle CSVs
SAVE_ARRAYS = True  # set True to write D/I/P/R arrays

# Cancer microsim or not (currently not functional)
USE_CANCER_MICROSIM = False  # set True to use microsim for cancer

# ---
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
TMATS_DIR = PROJECT_ROOT / "tmats"


# ------------------ helpers ------------------
def load_strategies(file: str | Path) -> list[strategy.Strategy]:
    return strategy.load_strategies(STRATEGIES_DIR / file)


def load_tmat_stack(path: Path) -> np.ndarray:
    """
    Load transition matrix stack.
    Supports:
      - 5D: [gene, sex, age_layer, to, from]  (recommended)
      - 4D: [sex, age_layer, to, from]        (single gene file)
    """
    path = PROJECT_ROOT / "tmats" / TMAT_STACK_FILE
    arr = np.load(path).astype(np.float64, copy=False)
    if arr.ndim not in (4, 5):
        raise ValueError(f"Unexpected TMAT shape {arr.shape}")
    return arr


def slice_tmat(stack: np.ndarray, gene: str, sex: str) -> np.ndarray:
    """Return (ages, from, to) tmat for specific gene and sex"""
    sex_idx = c.SEXES.index(sex)
    if stack.ndim == 5:
        gene_idx = c.GENES.index(gene) if gene in c.GENES else 0
        tmat = stack[sex_idx, gene_idx, :, :, :]
    else:  # 4D single-gene file
        tmat = stack[sex_idx, :, :, :]

    # Make cancer detected states absorb for post-hoc manipulation
    if USE_CANCER_MICROSIM:
        tmat[:, :, c.health_states_stoi["death_cancer"]] = 0
        tmat[:, c.detected_states_idx, c.detected_states_idx] = 1
    return tmat


# ------------------ model run ------------------


def run_one(
    strat: strategy.Strategy, co: cohort.Cohort
) -> Tuple[markov.MarkovModel, pd.DataFrame, pd.DataFrame]:
    """Runs model, computes outputs, and conditionally save CSVs"""

    m = markov.MarkovModel(strat, co)

    logging.info("Running %s ...", m)
    m.run_markov()

    if USE_CANCER_MICROSIM:
        cancer_microsim.run_cancer_microsim(m)

    per_cycle, lifetime = build_costs_qalys_tables(
        m,
        annual_discount_costs=0.03,
        annual_discount_qalys=0.03,
        use_half_cycle=False,
        state_util_weights=None,
    )

    if SAVE_LIFETIME_IND:
        save_model_outputs(
            m, lifetime, percycle=False, out_dir=OUTPUTS_DIR, suffix="lifetime"
        )
    if SAVE_PER_CYCLE:
        save_model_outputs(
            m, per_cycle, percycle=True, out_dir=OUTPUTS_DIR, suffix="per_cycle"
        )
    if SAVE_ARRAYS:
        save_run_arrays(m, OUTPUTS_DIR)

    return m, per_cycle, lifetime


def run_one_cohort_all_strategies(gene, sex, strategies, tmat_stack):
    """Run ALL strategies for ONE cohort (gene, sex) and compute ICERs within that cohort."""
    logging.info("=== Cohort: %s / %s ===", gene, sex)

    tmat = slice_tmat(tmat_stack, gene, sex)
    co = cohort.Cohort(gene, sex, tmat)

    results, lifetime_rows = [], []
    for strat in strategies:
        m, pc, lt = run_one(strat, co)
        results.append((str(m.strategy), pc, lt))

        row = lt.copy()
        row.insert(0, "strategy", str(m.strategy))
        row.insert(0, "sex", sex)
        row.insert(0, "gene", gene)
        lifetime_rows.append(row)

    # Concat lifetime data
    lifetime_all = pd.concat(lifetime_rows, ignore_index=True)

    # Add percent change in CRC incidence vs. no screening
    lifetime_all = add_crc_vs_nh(
        lifetime_all, base_strategy="NH", total_col="crc_incidents_total"
    )

    # Save outputs ----
    lifetime_all.to_csv(
        OUTPUTS_DIR / f"{co._safe_label}__lifetime_all_strategies.csv",
        index=False,
    )
    # Save summary
    save_lifetime_summary(lifetime_all, OUTPUTS_DIR, base_name=co._safe_label)

    # Get ICERs for all strategies ran within this cohort
    summary = summarize_for_icer(results, use_discounted=True)
    icer_table = compute_icers(summary)
    icer_table.to_csv(OUTPUTS_DIR / f"{co._safe_label}__icers.csv", index=False)


def run_one_strategy_all_genes(strat, tmat_stack):
    """Run single strategy for all genes and sexes; no ICERs calculated."""
    lifetime_rows = []
    for gene in GENES_TO_RUN:
        for sex in SEXES_TO_RUN:
            tmat = slice_tmat(tmat_stack, gene, sex)
            co = cohort.Cohort(gene, sex, tmat)
            m, pc, lt = run_one(strat, co)
            row = lt.copy()
            row.insert(0, "strategy", str(strat))
            row.insert(0, "sex", sex)
            row.insert(0, "gene", gene)
            lifetime_rows.append(row)

    lifetime_all = pd.concat(lifetime_rows, ignore_index=True)
    lifetime_all = add_crc_vs_nh(
        lifetime_all, base_strategy="NH", total_col="crc_incidents_total"
    )
    lifetime_all.to_csv(
        OUTPUTS_DIR / f"all_genes__{str(strat).replace(' ', '_')}__lifetime.csv",
        index=False,
    )
    save_lifetime_summary(lifetime_all, OUTPUTS_DIR, base_name="all_genes__")


# ------------------ main flow ------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load all strategies (gene-agnostic)
    strategies = load_strategies(STRATEGIES_FILE)
    if not strategies:
        raise ValueError("No strategies found. Check strategies file.")

    # Load TMATs
    tmat_stack = load_tmat_stack(TMATS_DIR / TMAT_STACK_FILE)

    # Make OUTPUTS_DIR if not already made
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_MODE == "single_gene_all_strategies":
        # For each gene/sex, run all strategies and compute ICERs WITHIN that cohort
        for gene in GENES_TO_RUN:
            for sex in SEXES_TO_RUN:
                run_one_cohort_all_strategies(gene, sex, strategies, tmat_stack)

    elif RUN_MODE == "single_strategy_all_genes":
        strat = strategies[0]  # NOTE: only first strategy is used in this mode
        run_one_strategy_all_genes(strat, tmat_stack)

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
