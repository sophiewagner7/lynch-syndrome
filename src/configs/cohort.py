import global_configs


class CohortSpec:
    def __init__(self, gene, sex, strategy, stop_age=75):
        self.gene = gene
        self.sex = sex
        self.screen_stop_age = stop_age
        self.strategy = strategy
        self.strategy_name = strategy.strategy_name
        self.screen_start_age = strategy.start_age_colo
        self.screening_cycles = self._get_screening_cycles()
        self.tmat = global_configs.tmat[gene][sex]

    def _get_screening_cycles(self):
        pass
