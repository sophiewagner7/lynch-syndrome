# strategy.py
"""
Load strategies and create class objects.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

from configs import global_configs as c

AltTestName = Literal["FIT", "sDNA", "SCED"]


@dataclass(frozen=True)
class TestSpec:
    interval_years: int
    start_age: int
    test: Literal["colo"] | AltTestName


@dataclass(frozen=True)
class Strategy:
    strategy_type: Literal["NH", "Colo", "Colo+Alt"]
    colo: Optional[TestSpec] = None
    alt_test: Optional[TestSpec] = None

    @property
    def is_NH(self) -> bool:
        return self.strategy_type == "NH"

    def __str__(self) -> str:
        """
        Short human-readable description (e.g., 'Colo Q2Y @ 25 + FIT Q1Y @ 36').
        """
        if self.is_NH:
            return "NH"
        parts: list[str] = []
        if self.colo:
            parts.append(f"Colo Q{self.colo.interval_years}Y @ {self.colo.start_age}")
        if self.alt_test:
            parts.append(
                f"+ {self.alt_test.test} Q{self.alt_test.interval_years}Y @ {self.alt_test.start_age}"
            )
        return " ".join(parts) if parts else self.strategy_type

    def _check_valid_inputs(self) -> None:
        """
        Validate against global config rules.
        """
        errors = []

        if self.strategy_type not in ["NH", "Colo", "Colo+Alt"]:
            errors.append(
                f"Invalid strategy_type '{self.strategy_type}'. "
                f"Allowed: ['NH', 'Colo', 'Colo+Alt']"
            )

        if self.colo:
            if self.colo.interval_years not in c.SCREENING_INTERVALS:
                errors.append(
                    f"Invalid colo interval '{self.colo.interval_years}'. "
                    f"Allowed: {c.SCREENING_INTERVALS}"
                )
            if self.colo.start_age not in c.SCREENING_START_AGES:
                errors.append(
                    f"Invalid colo start_age '{self.colo.start_age}'. "
                    f"Allowed: {c.SCREENING_START_AGES}"
                )

        if self.alt_test:
            if self.alt_test.test not in c.ALT_TESTS:
                errors.append(
                    f"Invalid alt test '{self.alt_test.test}'. "
                    f"Allowed: {c.ALT_TESTS}"
                )
            if self.alt_test.interval_years not in c.SCREENING_INTERVALS:
                errors.append(
                    f"Invalid alt interval '{self.alt_test.interval_years}'. "
                    f"Allowed: {c.SCREENING_INTERVALS}"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def tests(self) -> list[str]:
        """Return list of test names used in this strategy."""
        out = []
        if self.colo:
            out.append("colo")
        if self.alt_test:
            out.append(self.alt_test.test)
        return out


# --- Load strategies.json ---
def load_strategies(path: str | Path) -> list[Strategy]:
    """
    Load strategies from a JSON file with shape:
    {
      "strategies": [
        { "id": "...", "strategy_type": "...", "colo": {...}, "alt_test": {...} },
        ...
      ]
    }
    Returns a list[Strategy].
    """
    data = json.loads(Path(path).read_text())

    # Support both the documented shape and a bare list for convenience
    strategy_items = data.get("strategies", data)

    strategies: list[Strategy] = []
    for s in strategy_items:
        colo = s.get("colo")
        alt = s.get("alt_test")
        strat = Strategy(
            strategy_type=s["strategy_type"],
            colo=TestSpec(**colo) if colo else None,
            alt_test=TestSpec(**alt) if alt else None,
        )
        strat._check_valid_inputs()
        strategies.append(strat)

    return strategies
