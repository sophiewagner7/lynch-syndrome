import logging

# TODO: Make this read in from csv/xlsx
# TODO: Make test performance optional input in main.py
# Screening test specifications
SCREENING_TEST_SPECS = {
    "FIT": {
        "sens_LRpolyp": 0.1,
        "sens_HRpolyp": 0.2,
        "sens_stage_1": 0.79,
        "sens_stage_2": 0.79,
        "sens_stage_3": 0.79,
        "sens_stage_4": 0.79,
        "spec": 0.95,
        "fpr": 0.05,
        "c_test": 20,
        "du_test": 0,
        "p_complication": 0,
        "du_complication": 0,
        "p_death_complication": 0,
        "c_complication": 0,
    },
    "Colo": {
        "sens_LRpolyp": 0.1,
        "sens_HRpolyp": 0.2,
        "sens_stage_1": 0.79,
        "sens_stage_2": 0.79,
        "sens_stage_3": 0.79,
        "sens_stage_4": 0.79,
        "spec": 0.95,
        "fpr": 0.05,
        "c_test": 20,
        "du_test": 0,
        "p_complication": 0,
        "du_complication": 0,
        "p_death_complication": 0,
        "c_complication": 0,
    },
    "sDNA": {
        "sens_LRpolyp": 0.1,
        "sens_HRpolyp": 0.2,
        "sens_stage_1": 0.79,
        "sens_stage_2": 0.79,
        "sens_stage_3": 0.79,
        "sens_stage_4": 0.79,
        "spec": 0.95,
        "fpr": 0.05,
        "c_test": 20,
        "du_test": 0,
        "p_complication": 0,
        "du_complication": 0,
        "p_death_complication": 0,
        "c_complication": 0,
    },
}


# STRAT TYPES:  NH, Colo only, Colo + FIT, Colo + sDNA
class Strategy:
    def __init__(self, gene, strategy, strategy_type, stool_test=None):
        self.gene = gene
        self.strategy = strategy
        self.strategy_type = strategy_type
        self.start_age_colo = strategy[1]
        self.interval_colo = strategy[0]
        self.start_age_stool = strategy[3] if stool_test else None
        self.interval_stool = strategy[2] if stool_test else None
        self.stool_test = stool_test
        self.is_NH = strategy_type == "NH"
        self.strategy_name = self._get_strategy_name()

    def _get_strategy_name(self):
        if self.is_NH:
            strategy_name = "NH"
        elif self.stool_test:
            strategy_name = f"Colo Q{self.interval_colo}Y {self.stool_test} Q{self.interval_stool}, Start Age: {self.start_age_colo}"
        elif self.stool_test is None:
            strategy_name = (
                f"Colo Q{self.interval_colo}Y, Start Age: {self.start_age_colo}"
            )
        else:
            logging.error("Strategy type not recognized")
            strategy_name = "UNKNOWN"
        return strategy_name

    def __repr__(self):
        return f"Strategy(gene={self.gene}, stool_test={self.stool_test}, strategy={self.strategy})"


INTERDIGITATION_STRATS = {
    "MLH1": [[2, 25, 1, 26], [4, 25, 2, 26], [2, 30, 1, 31], [4, 30, 2, 31]],
    "MSH2": [[2, 25, 1, 26], [4, 25, 2, 26], [2, 30, 1, 31], [4, 30, 2, 31]],
    "MSH6": [[2, 25, 1, 26], [4, 25, 2, 26], [2, 30, 1, 31], [4, 30, 2, 31]],
    "PMS2": [[2, 25, 1, 26], [4, 25, 2, 26], [2, 35, 1, 36], [4, 30, 2, 31]],
}
NONINTERDIGITATION_STRATS = {
    "MLH1": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH2": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH6": [[5, 40], [3, 40], [3, 35], [1, 25]],
    "PMS2": [[5, 40], [4, 40], [3, 40], [3, 35], [1, 25]],
}
NH_STRATS = {
    "MLH1": [[0, 25]],
    "MSH2": [[0, 25]],
    "MSH6": [[0, 25]],
    "PMS2": [[0, 25]],
}
# Create dictionary of strategies
strategies = {}
# Add NH strategies
for gene in NH_STRATS:
    for strat in NH_STRATS[gene]:
        temp = Strategy(gene=gene, strategy=strat, strategy_type="NH")
        strategies[f"{gene} {temp.strategy_name}"] = temp
# Add non-interdigitated strategies
for gene in NONINTERDIGITATION_STRATS:
    for strat in NONINTERDIGITATION_STRATS[gene]:
        temp = Strategy(gene=gene, strategy=strat, strategy_type="Colo")
        strategies[f"{gene} {temp.strategy_name}"] = temp
# Add interdigitated strategies
for test in ["FIT", "sDNA"]:
    for gene in INTERDIGITATION_STRATS:
        for strat in INTERDIGITATION_STRATS[gene]:
            temp = Strategy(
                gene=gene, strategy=strat, strategy_type=f"Colo{test}", stool_test=test
            )
            strategies[f"{gene} {temp.strategy_name}"] = temp
