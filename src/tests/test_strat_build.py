# test_strat_build.py
"""
Test that the way that we initialize our Strategy and RunSpec class, loading from json works.
To use this test, cd into src, and run with:
python3 -m tests.test_strat_build
"""

from pathlib import Path

import numpy as np

from configs import strategy, c
from model import cohort, markov

# ------------------ paths ------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
STRATEGIES_DIR = SRC_DIR / "tests"
TMATS_DIR = PROJECT_ROOT / "tmats"

STRATEGIES_FILE = "test_strats.json"
TMAT_STACK_FILE = (
    "PMS2_20250922_1040.npy"  # recommended: full [gene, sex, age, to, from] stack
)


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
        gene_idx = 0
        return stack[sex_idx, gene_idx, :, :, :]
    # 4D file → single gene: shape [sex, age, to, from]
    return stack[sex_idx, :, :, :]


def main():
    strats = load_strategies(STRATEGIES_DIR / STRATEGIES_FILE)
    tmat_stack = load_tmat_stack(TMATS_DIR / TMAT_STACK_FILE)
    gene, sex = "MLH1", "male"
    tmat = slice_tmat(tmat_stack, gene, sex)
    co = cohort.Cohort(gene, sex, tmat)
    for strat in strats:
        print(strat)
        rs = markov.RunSpec(strat, co)
        protocol = rs.get_screening_protocol()
        assert len(protocol) == c.NUM_CYCLES, "protocol length check"
        print(protocol[::12])
        if strat.annual:
            assert (
                protocol[(strat.annual.start_age - c.START_AGE) * 12]
                == strat.annual.test_seq[0]
            ), "first test check"
        if strat.strategy_type == "NH":
            assert all(x is None for x in protocol), "NH check"
        if strat.strategy_type == "Colo":
            if strat.colo is not None:
                assert all(
                    x == "colo"
                    for x in protocol[
                        (strat.colo.start_age - c.START_AGE)
                        * 12 : (c.SCREENING_STOP_AGE - c.START_AGE)
                        * 12 : strat.colo.interval_years
                        * 12
                    ]
                ), "colo only check"
        assert all(
            x is None for x in protocol[c.SCREENING_STOP_AGE * 12 :]
        ), "stop age check"


main()
