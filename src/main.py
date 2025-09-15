# main.py
"""
Run Markov model(s) based on strategy and cohort inputs.

Run modes:
- "by_gene_strategies": for each (gene, sex) run every strategy; compute ICERs within each cohort.
- "by_strategy_all_genes": pick one strategy and run it across multiple genes/sexes; no ICERs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from configs import global_configs as c
from configs import inputs, strategy
from model import cohort, markov
from processing.single_model_outputs import (
    build_costs_qalys_tables,
    save_run_costs_utils,  # assumes you have this; otherwise swap to your save function
)
from processing.icer import compute_icers, summarize_for_icer


# ------------------ paths ------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "20250915"
TMATS_DIR = PROJECT_ROOT / "tmats"

# ---************ USER INPUTS *************---
# Strategies file: list of Strategy objects (no gene baked in)
STRATEGIES_FILE = "mlh1_strats.json"
TMAT_STACK_FILE = (
    "MLH1_20250915_1121.npy"  # recommended: full [gene, sex, age, to, from] stack
)

# Select run mode:
RUN_MODE = (
    "by_gene_strategies"  # OPTIONS: "by_strategy_all_genes", "by_gene_strategies"
)

# Cohort selections:
GENES_TO_RUN = ["MLH1"]  # options from c.GENES, e.g. ["MLH1","MSH2","MSH6","PMS2"]
SEXES_TO_RUN = ["male", "female"]  # ["male","female"]

# If RUN_MODE == "by_strategy_all_genes", choose ONE strategy by index (0-based) or name string
SELECTED_STRATEGY = 0  # index into loaded strategies list


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
    arr = np.load(path).astype(np.float64, copy=False)
    print(arr.shape)
    if arr.ndim not in (4, 5):
        raise ValueError(f"Unexpected TMAT shape {arr.shape}; expected 4D or 5D.")
    return arr


def slice_tmat(
    stack: np.ndarray,
    gene: str,
    sex: str,
) -> np.ndarray:
    """
    Return [age_layer, to, from] for a (gene, sex).
    """
    sex_idx = c.SEXES.index(sex)
    if stack.ndim == 5:
        gene_idx = c.GENES.index(gene)
        return stack[sex_idx, gene_idx, :, :, :]
    # 4D file → single gene: shape [sex, age, to, from]
    return stack[sex_idx, :, :, :]


def run_one(
    strat: strategy.Strategy,
    gene: str,
    sex: str,
    tmat_stack: np.ndarray,
) -> Tuple[markov.MarkovModel, pd.DataFrame, pd.DataFrame]:
    """
    Build cohort, run model, compute outputs, and save CSVs.
    """
    tmat = slice_tmat(tmat_stack, gene, sex)
    co = cohort.Cohort(gene, sex, tmat)
    rs = markov.RunSpec(strat, co)

    m = markov.MarkovModel(rs)
    logging.info("Running %s ...", rs)
    m.run_markov()

    per_cycle, lifetime = build_costs_qalys_tables(
        m,
        annual_discount_costs=0.03,
        annual_discount_qalys=0.03,
        use_half_cycle=False,
        state_util_weights=None,
    )
    save_run_costs_utils(m, per_cycle, lifetime, OUTPUTS_DIR)
    return m, per_cycle, lifetime


def summarize_cancers(model: markov.MarkovModel) -> float:
    """
    Quick total CRC count (sum of inflows to u/d-stage columns if your indices 7:11
    are cancer states; adjust as needed).
    """
    return float(model.I[:, 7:11].sum())


# ------------------ main flow ------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load all strategies (gene-agnostic)
    strategies = load_strategies(STRATEGIES_FILE)
    if not strategies:
        raise ValueError("No strategies found. Check strategies file.")

    # Load TMATs
    tmat_stack = load_tmat_stack(TMATS_DIR / TMAT_STACK_FILE)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_MODE == "by_gene_strategies":
        # For each gene/sex, run all strategies and compute ICERs WITHIN that cohort
        for gene in GENES_TO_RUN:
            for sex in SEXES_TO_RUN:
                logging.info("=== Cohort: %s / %s ===", gene, sex)

                models = []
                cancer_counts = {}

                for strat in strategies:
                    model, per_cycle, lifetime = run_one(strat, gene, sex, tmat_stack)
                    models.append(
                        (str(model.spec.strategy), model, per_cycle, lifetime)
                    )
                    cancer_counts[str(model.spec.strategy)] = summarize_cancers(model)

                # Save quick cancer summary
                cancer_df = (
                    pd.DataFrame(
                        {
                            "strategy": list(cancer_counts.keys()),
                            "total_crc": list(cancer_counts.values()),
                        }
                    )
                    .sort_values("strategy")
                    .reset_index(drop=True)
                )
                co = models[0][1].spec.cohort  # any model to get clean label
                cancer_df.to_csv(
                    OUTPUTS_DIR / f"{co._safe_label}__total_cancers.csv", index=False
                )

                # ICERs for strategies within THIS gene/sex only
                results = []
                for lbl, m, pc, lt in models:
                    results.append((lbl, pc, lt))
                summary = summarize_for_icer(results, use_discounted=True)
                icer_table = compute_icers(summary)
                icer_table.to_csv(
                    OUTPUTS_DIR / f"{co._safe_label}__icers.csv", index=False
                )

    elif RUN_MODE == "by_strategy_all_genes":
        # Choose a single strategy (index or name)
        if isinstance(SELECTED_STRATEGY, int):
            strat = strategies[SELECTED_STRATEGY]
        else:
            # name/text match against __str__()
            matched = [s for s in strategies if str(s) == SELECTED_STRATEGY]
            if not matched:
                raise ValueError(
                    f"Strategy '{SELECTED_STRATEGY}' not found in loaded strategies."
                )
            strat = matched[0]

        summary_rows = []
        for gene in GENES_TO_RUN:
            for sex in SEXES_TO_RUN:
                model, per_cycle, lifetime = run_one(strat, gene, sex, tmat_stack)
                co = model.spec.cohort
                # Collect a compact summary line (discounted totals are typical for comparisons)
                summary_rows.append(
                    {
                        "gene": gene,
                        "sex": sex,
                        "strategy": str(model.spec.strategy),
                        "total_crc": summarize_cancers(model),
                        "cost_disc": float(lifetime["total_cost_disc"].iloc[0]),
                        "qalys_disc": float(lifetime["total_qalys_disc"].iloc[0]),
                        "ly_total": float(lifetime["total_ly"].iloc[0]),
                    }
                )

        # No ICERs across different genes (not comparable cohorts)
        pd.DataFrame(summary_rows).to_csv(
            OUTPUTS_DIR / f"all_genes__{str(strat).replace(' ', '_')}__summary.csv",
            index=False,
        )
        logging.info("Saved all-genes summary for strategy: %s", strat)

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
