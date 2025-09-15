import json
from pathlib import Path

import pandas as pd
import numpy as np

import configs.global_configs as c

# ---------------------------------------------------------------------------- #
# DEFINE PATHS
# ---------------------------------------------------------------------------- #

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]  # .../project_root
SRC_DIR = HERE.parents[1]
CONFIGS_DIR = HERE.parents[0]
DATA_DIR = PROJECT_ROOT / "data"  # .../project_root/data
LT_DIR = DATA_DIR / "lifetables"
TARGET_DIR = DATA_DIR / "targets"
TMAT_DIR = PROJECT_ROOT / "tmats"

# ---------------------------------------------------------------------------- #
# LIFE TABLES
# ---------------------------------------------------------------------------- #

# Life table with 5-year intervals
lt_5y = pd.read_csv(LT_DIR / "lifetable_5y.csv", index_col=0)

# Life table with 1-year intervals
lt_1y = pd.read_csv(LT_DIR / "lifetable_1y.csv", index_col=0)

# ---------------------------------------------------------------------------- #
# CALIBRATION TARGETS
# ---------------------------------------------------------------------------- #

# CRC incidence ---------------------------------------------------------------
# CRC incidence targets by gene and sex (age-specific, annual incidence)
# MultiIndex columns: (gene, sex) — e.g., incidence_target['MLH1']['female']
incidence_target = pd.read_csv(
    TARGET_DIR / "crc_incidence_1y_to85.csv", index_col=0, header=[0, 1]
)
# Example: incidence_target['MLH1']['male'] → incidence by age

# CRC stage distribution at diagnosis ----------------------------------------
# CRC stage distribution at diagnosis (proportion of cases by stage)
# DataFrame columns: ['stage', 'value']
stage_dist_target = pd.read_csv(TARGET_DIR / "stage_distribution.csv", index_col=0)
stage_dist_target_dict = stage_dist_target["value"].to_dict()
# Example: stage_dist_target_dict['stage_1'] → 0.40

# Polyp prevalence -----------------------------------------------------------
# Polyp prevalence targets by gene (cumulative incidence by age 60)
# DataFrame columns: ['gene', 'value']
polyp_target = pd.read_csv(TARGET_DIR / "adenoma_risk_age60.csv", index_col=0)
polyp_targets_dict = polyp_target["value"].to_dict()
# Example: polyp_targets_dict['MSH2'] → 0.45

# ---------------------------------------------------------------------------- #
# SCREENING TEST SPECS
# ---------------------------------------------------------------------------- #

with open(CONFIGS_DIR / "screening_test_specs.json", "r") as f:
    SCREENING_TEST_SPECS = json.load(f)
