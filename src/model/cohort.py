from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Cohort:
    gene: str
    sex: str
    # tmat[age_layer, from_state, to_state]
    tmat: np.ndarray

    @property
    def _safe_label(self):
        return f"{self.gene}_{self.sex}"

    def __str__(self) -> str:
        return f"{self.gene} {self.sex}"
