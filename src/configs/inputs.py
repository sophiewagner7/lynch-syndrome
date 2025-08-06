import os

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
LT_DIR = os.path.join(DATA_DIR, "lifetables")

### Inputs

# Load life tables
lt_5y = pd.read_csv(os.path.join(LT_DIR, "lifetable_5y.csv"), index_col=0)
lt_1y = pd.read_csv(os.path.join(LT_DIR, "lifetable_1y.csv"), index_col=0)

### Targets

# Load CRC incidence data
incidence_target = pd.read_csv(
    os.path.join(
        DATA_DIR,
    )
)

# Load polyp prevalence targets
polyp_targets = pd.read_csv(
    os.path.join(DATA_DIR, "mecklin_polyp_risk_1y.csv"), index_col=0
)
