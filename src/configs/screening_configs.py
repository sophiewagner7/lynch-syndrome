import logging
import json

import numpy as np

import inputs as i
import global_configs as c

# ---------------------------------------------------------------------------- #
# SCREENING SPECS
# ---------------------------------------------------------------------------- #
with open(i.CONFIGS_DIR / "screening_test_specs.json", "r") as f:
    SCREENING_TEST_SPECS = json.load(f)

# ---------------------------------------------------------------------------- #
# SCREENING STRATEGIES
# ---------------------------------------------------------------------------- #
with open(i.CONFIGS_DIR / "screening_strategies.json", "r") as f:
    SCREENING_STRATEGIES = json.load(f)

NH_STRATS = SCREENING_STRATEGIES["NH_STRATS"]
INTERDIGITATION_STRATS = SCREENING_STRATEGIES["INTERDIGITATION_STRATS"]
NONINTERDIGITATION_STRATS = SCREENING_STRATEGIES["NONINTERDIGITATION_STRATS"]


# ---------------------------------------------------------------------------- #
# STRATEGY CLASS
# ---------------------------------------------------------------------------- #
class Strategy:
    def __init__(self, gene, strategy_type, colo=None, alt_test=None):
        self.gene = gene
        self.strategy_type = strategy_type  # NH, Colo, or Colo+Alt
        self.colo = colo  # dict like {"interval": 2, "start_age": 25}
        self.alt_test = (
            alt_test  # dict like {"interval": 1, "start_age": 26, "test": "FIT"}
        )
        self.is_NH = strategy_type == "NH"
        self.check_valid_inputs()

    def __str__(self):
        """
        Return summary representation of strategy specs. This is returned with print(s).
        """
        if self.is_NH:
            return f"{self.gene}: NH"
        out = [f"{self.gene}:"]
        if self.colo:
            out.append(f"Colo Q{self.colo['interval']}Y @ {self.colo['start_age']}")
        if self.alt_test:
            out.append(
                f"+ {self.alt_test['test']} Q{self.alt_test['interval']}Y @ {self.alt_test['start_age']}"
            )
        return " ".join(out)

    def __repr__(self):
        """
        Developer/debug representation — should look like valid Python code
        to recreate the object. Returns this if you just type strategy in console or .py.
        """
        return (
            f"Strategy(gene={self.gene!r}, strategy_type={self.strategy_type!r}, "
            f"colo={self.colo!r}, alt_test={self.alt_test!r})"
        )

    def check_valid_inputs(self):
        """
        Check that the inputs for creating the strategy are valid according to our
        preset rules from the global_configs.py file.
        """
        errors = []

        if self.gene not in c.GENES:
            errors.append(
                f"Invalid gene '{self.gene}' in strategy '{self}'. "
                f"Allowed: {c.GENES}"
            )

        if self.strategy_type not in ["NH", "Colo", "Colo+Alt"]:
            errors.append(
                f"Invalid strategy_type '{self.strategy_type}' for strategy '{self.strategy_type}'. "
                f"Allowed: ['NH', 'Colo', 'Colo+Alt']"
            )

        if self.colo:
            if self.colo["interval"] not in c.SCREENING_INTERVALS:
                errors.append(
                    f"Invalid colonoscopy interval '{self.colo['interval']}' in strategy '{self.strategy_type}'. "
                    f"Allowed: {c.SCREENING_INTERVALS}"
                )
            if self.colo["start_age"] not in c.SCREENING_START_AGES:
                errors.append(
                    f"Invalid colonoscopy start_age '{self.colo['start_age']}' in strategy '{self.strategy_type}'. "
                    f"Allowed: {c.SCREENING_START_AGES}"
                )

        if self.alt_test:
            if self.alt_test["test"] not in c.ALT_TESTS:
                errors.append(
                    f"Invalid alternative test '{self.alt_test['test']}' in strategy '{self.strategy_type}'. "
                    f"Allowed: {c.ALT_TESTS}"
                )
            if self.alt_test["interval"] not in c.SCREENING_INTERVALS:
                errors.append(
                    f"Invalid alternative test interval '{self.alt_test['interval']}' in strategy '{self.strategy_type}'. "
                    f"Allowed: {c.SCREENING_INTERVALS}"
                )
            # TODO: define proper screening start ages for our alternative tests

        if errors:
            raise ValueError("\n".join(errors))

    def create_events(self):
        if self.is_NH:
            return []
        if self.colo and not self.alt_test:
            events = np.arange(
                self.colo["start_age"], c.END_AGE, step=self.colo["interval"]
            )
            return events


# ---------------------------------------------------------------------------- #
# CREATE STRATEGY OBJECTS FROM JSON
# ---------------------------------------------------------------------------- #
strategies = []

# NH
for gene, lst in NH_STRATS.items():
    for _ in lst:
        strategies.append(Strategy(gene, "NH"))

# Colo only
for gene, lst in NONINTERDIGITATION_STRATS.items():
    for s in lst:
        strategies.append(Strategy(gene, "Colo", colo=s["colo"]))

# Interdigitation
for gene, lst in INTERDIGITATION_STRATS.items():
    for s in lst:
        strategies.append(
            Strategy(gene, "Colo+Alt", colo=s["colo"], alt_test=s["alt_test"])
        )

# Print them
for s in strategies:
    print(s)
