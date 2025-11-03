# strategy.py
"""
Load strategies and create class objects.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, List, Dict, Any, cast

import numpy as np

from configs import global_configs as c

TestName = Literal["colo", "FIT", "sDNA", "SCED"]


@dataclass(frozen=True)
class Strategy:
    strategy_type: Literal["NH", "Colo", "Alt", "Colo+Alt"]

    # flattened colonoscopy fields
    colo_interval_years: Optional[int] = None
    colo_start_age: Optional[int] = None

    # flattened alternate test fields
    alt_test: Optional[TestName] = None
    alt_interval_years: Optional[int] = None
    alt_start_age: Optional[int] = None

    # flattened annual pattern
    annual_start_age: Optional[int] = None
    annual_seq: Optional[List[Optional[TestName]]] = None

    def __post_init__(self):
        self._validate_inputs()

    @property
    def is_NH(self) -> bool:
        return self.strategy_type == "NH"

    @property
    def screen_start_age(self):
        """Return earliest screening start age or END_AGE+1 if no screening"""
        if self.is_NH:
            return c.END_AGE + 1  # no screening
        if self.annual_seq is not None and self.annual_start_age is not None:
            return cast(int, self.annual_start_age)
        else:
            colo_start = (
                cast(int, self.colo_start_age) if self.colo_start_age else c.END_AGE + 1
            )
            alt_start = (
                cast(int, self.alt_start_age) if self.alt_test else c.END_AGE + 1
            )
            return min(colo_start, alt_start)

    @property
    def screen_stop_age(self):
        return c.SCREENING_STOP_AGE

    def __str__(self) -> str:
        if self.is_NH:
            return "NH"

        # Annual pattern takes precedence
        if self.annual_seq is not None:
            pretty = "-".join(
                ("None" if t is None else ("FIT" if t == "FIT" else t.capitalize()))
                for t in self.annual_seq
            )
            return f"{pretty} @ {self.annual_start_age}"

        # Otherwise, interval description(s)
        parts: List[str] = []
        if self.colo_interval_years and self.colo_start_age is not None:
            parts.append(f"Colo Q{self.colo_interval_years}Y @ {self.colo_start_age}")
        if self.alt_test and self.alt_interval_years and self.alt_start_age is not None:
            lead = "+ " if parts else ""
            parts.append(
                f"{lead}{self.alt_test} Q{self.alt_interval_years}Y @ {self.alt_start_age}"
            )
        return " ".join(parts) if parts else self.strategy_type

    def _validate_inputs(self):
        """Validate inputs from json"""
        errs = []
        if (
            self.strategy_type != "Colo+Alt"
            and self.strategy_type != "NH"
            and self.strategy_type != "Colo"
            and self.strategy_type != "Alt"
        ):
            errs.append(f"Unsupported strategy_type: {self.strategy_type}")

        if self.strategy_type == "Colo+Alt":
            if not (self.annual_seq or self.colo_interval_years or self.alt_test):
                errs.append(
                    "Colo+Alt requires either annual pattern or at least one interval test."
                )
            if self.annual_seq:
                if all(x is None for x in self.annual_seq):
                    errs.append("annual.items must include at least one non-None test.")
                for x in self.annual_seq:
                    if x is not None and x not in ("colo", "FIT", "sDNA", "SCED"):
                        errs.append(f"Invalid annual test: {x}")

        if errs:
            raise ValueError("\n".join(errs))

    def tests(self) -> list[str]:
        """Return list of test names used in this strategy."""
        out = []
        if self.annual_seq is not None:
            test_set = set(self.annual_seq)
            test_set.discard(None)
            while test_set:
                out.append(test_set.pop())
        else:
            if self.colo_start_age is not None:
                out.append("colo")
            if self.alt_test is not None:
                out.append(self.alt_test)

        return out

    def get_screening_protocol(self) -> List[Optional[str]]:
        """
        Generate screening protocol based on strategy specifications.
        Assumes monthly cycles.
        Returns a list of length equal to number of cycles with screening tests name or None.
        [None, None, ..., "colo",...] (length = number of cycles)
        """
        # Initialize protocol list with None (no screening)
        protocol: List[Optional[str]] = [None] * c.NUM_CYCLES

        # No screening for NH strategy
        if self.is_NH:
            return protocol

        if self.annual_seq is not None and self.annual_start_age is not None:
            pattern = self.annual_seq
            repeat = len(pattern)
            screen_start_age = self.annual_start_age
            i = 0
            for age in np.arange(screen_start_age, self.screen_stop_age + 1, 1):
                cycle = (age - c.START_AGE) * 12
                if i == repeat:
                    i = 0
                if 0 <= cycle < c.NUM_CYCLES:
                    protocol[cycle] = pattern[i]
                    i += 1
        else:
            # Add colonoscopy if specified
            if self.colo_interval_years and self.colo_start_age is not None:
                interval = self.colo_interval_years
                screen_start_age = self.colo_start_age
                for age in np.arange(
                    screen_start_age, self.screen_stop_age + 1, interval
                ):
                    cycle = (age - c.START_AGE) * 12
                    if 0 <= cycle < c.NUM_CYCLES:
                        protocol[cycle] = "colo"

            # Add alternative test if specified
            if self.alt_test:
                interval = self.alt_interval_years
                screen_start_age = self.alt_start_age
                test = self.alt_test
                for age in np.arange(
                    screen_start_age, self.screen_stop_age + 1, interval
                ):
                    cycle = (age - c.START_AGE) * 12
                    if 0 <= cycle < c.NUM_CYCLES and protocol[cycle] is None:
                        protocol[cycle] = test

        return protocol


# ---------- JSON loading  ----------
def _from_json_item(s: Dict[str, Any]) -> Strategy:
    # start with flat keys if provided
    kwargs: Dict[str, Any] = {
        "strategy_type": s["strategy_type"],
        "colo_interval_years": s.get("colo_interval_years"),
        "colo_start_age": s.get("colo_start_age"),
        "alt_test": s.get("alt_test") if isinstance(s.get("alt_test"), str) else None,
        "alt_interval_years": s.get("alt_interval_years"),
        "alt_start_age": s.get("alt_start_age"),
        "annual_start_age": s.get("annual_start_age"),
        "annual_seq": s.get("annual_seq"),
    }
    # nested sections (old shape) override if present
    if isinstance(s.get("colo"), dict):
        kwargs["colo_interval_years"] = s["colo"].get("interval_years")
        kwargs["colo_start_age"] = s["colo"].get("start_age")
    if isinstance(s.get("alt_test"), dict):
        kwargs["alt_test"] = s["alt_test"].get("test")
        kwargs["alt_interval_years"] = s["alt_test"].get("interval_years")
        kwargs["alt_start_age"] = s["alt_test"].get("start_age")
    if isinstance(s.get("annual"), dict):
        kwargs["annual_start_age"] = s["annual"].get("start_age")
        kwargs["annual_seq"] = s["annual"].get("test_seq")
    return Strategy(**kwargs)


def load_strategies(path: str | Path) -> List[Strategy]:
    data = json.loads(Path(path).read_text())
    items = data.get("strategies", data)
    if not isinstance(items, list):
        raise ValueError("Invalid JSON: expected a list or a 'strategies' list.")
    return [_from_json_item(s) for s in items]
