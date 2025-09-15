from configs.strategy import Strategy, TestSpec
from model.markov import MarkovModel
from model.cohort import Cohort
from model.runspec import RunSpec
from configs import inputs, c

import numpy as np

colo_test = TestSpec(1, 25, "colo")
strat = Strategy("Colo", colo=colo_test)
tmat = np.load(inputs.TMAT_DIR / "MLH1_20250825_1034.npy")
tmat_m = tmat[0, 0, :, :, :]
tmat_f = tmat[1, 0, :, :, :]
cohort = Cohort("MLH1", "male", tmat_m)
runspec = RunSpec(strat, cohort)

model = MarkovModel(runspec)
model.run_markov()


def test_protocol_placement_matches_spec():
    """
    Check that RunSpec creates the expected colonoscopy schedule.
    """
    gene = "MLH1"
    colo = TestSpec(interval_years=1, start_age=25, test="colo")
    strat = Strategy(strategy_type="Colo", colo=colo)

    # Minimal tmat for protocol test: 1 age_layer, identity Markov (no movement).
    S = c.n_states
    tmat_identity = np.zeros((1, S, S))
    np.fill_diagonal(tmat_identity[0], 1.0)

    cohort = Cohort(gene=gene, sex="male", tmat=tmat_identity)
    runspec = RunSpec(strategy=strat, cohort=cohort)

    protocol = runspec.get_screening_protocol()
    assert len(protocol) == runspec.cycles

    first_cycle = int((25 - runspec.model_start_age) * 12)
    # Annual: expect colo at first_cycle, first_cycle+12, ...
    assert protocol[first_cycle] == "colo"
    assert protocol[first_cycle + 12] == "colo"
    # And something before start_age should be None
    assert protocol[max(first_cycle - 1, 0)] is None
