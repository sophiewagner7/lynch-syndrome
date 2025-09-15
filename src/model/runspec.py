from dataclasses import dataclass
from typing import Optional, List, cast

import numpy as np

from configs.strategy import Strategy
from model.cohort import Cohort
from configs import global_configs as c


@dataclass
class RunSpec:
    strategy: Strategy
    cohort: Cohort

    @property
    def gene(self):
        return self.cohort.gene

    @property
    def tmat(self):
        return self.cohort.tmat

    @property
    def sex(self):
        return self.cohort.sex

    @property
    def cycles(self):
        return c.NUM_CYCLES

    @property
    def model_start_age(self):
        return c.START_AGE

    @property
    def screen_stop_age(self):
        return c.END_AGE

    @property
    def is_NH(self):
        return self.strategy.is_NH

    def __str__(self) -> str:
        """
        Return summary representation of strategy specs. This is returned with print(s).
        """
        return f"{self.gene}: {self.strategy}"

    def get_screening_protocol(self) -> List[Optional[str]]:
        """
        Generate screening protocol based on strategy specifications.
        Assumes monthly cycles.
        Returns a list of length equal to number of cycles with screening tests name or None.
        """
        # Initialize protocol list with None (no screening)
        protocol: List[Optional[str]] = [None] * self.cycles

        # No screening for NH strategy
        if self.is_NH:
            return protocol

        # Add colonsocopy if specified
        if self.strategy.colo:
            interval = self.strategy.colo.interval_years
            screen_start_age = self.strategy.colo.start_age
            for age in np.arange(screen_start_age, self.screen_stop_age + 1, interval):
                cycle = (age - self.model_start_age) * 12
                if 0 <= cycle < self.cycles:
                    protocol[cycle] = "colo"

        # Add alternative test if specified
        if self.strategy.alt_test:
            interval = self.strategy.alt_test.interval_years
            screen_start_age = self.strategy.alt_test.start_age
            test = self.strategy.alt_test.test
            for age in np.arange(screen_start_age, self.screen_stop_age + 1, interval):
                cycle = (age - self.model_start_age) * 12
                if 0 <= cycle < self.cycles:
                    protocol[cycle] = test

        return protocol
