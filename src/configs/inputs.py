import json
from pathlib import Path

import pandas as pd
import numpy as np

from configs import global_configs as c

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

# CRC cumulative incidence to age 75 ------------------------------------------
# By: gene, sex, age
# Source: Dominguez (PLSD)
incidence_cumulative_target = pd.read_csv(
    TARGET_DIR / "crc_incidence_cumulative_to_75.csv", index_col=["gene", "sex"]
)

# CRC cumulative incidence (CURVE, diff between above vals) to age 75 --------
# Source: Dominguez (PLSD)
incidence_cumulative_curve_target = pd.read_csv(
    TARGET_DIR / "crc_incidence_curve_to_75.csv", index_col=["gene", "sex"]
)

# CRC incidence by age 65 ----------------------------------------------------
# By: gene, sex
# Source: Dominguez (PLSD)
incidence_by_65_target = pd.read_csv(
    TARGET_DIR / "crc_incidence_by_65.csv", index_col=["gene", "sex"]
)

# CRC incidence by age 70 ----------------------------------------------------
# By: gene, sex
# Source: Wang / NCCN
incidence_by_70_target = pd.read_csv(
    TARGET_DIR / "crc_incidence_by_70.csv", index_col=["gene", "sex"]
)

# CRC incidence by age 80 ----------------------------------------------------
# By: gene
# Source: NCCN
incidence_by_80_target = pd.read_csv(
    TARGET_DIR / "crc_incidence_by_80.csv", index_col="gene"
)
incidence_by_80_target["lower"] /= 100.0
incidence_by_80_target["upper"] /= 100.0

# CRC median age at diagnosis ------------------------------------------------
# By: gene
# Source: NCCN
median_age_at_dx_target = pd.read_csv(
    TARGET_DIR / "crc_median_age.csv", index_col="gene"
)

# CRC stage distribution at diagnosis ----------------------------------------

# CRC stage distribution from SEER -------------------------------------------
# Average risk population stage distribution used
# Source: SEER
stage_dist_target = pd.read_csv(
    TARGET_DIR / "stage_distribution.csv", index_col="stage"
)

# Polyp prevalence -----------------------------------------------------------

# Polyp incidence by age 65 --------------------------------------------------
# By: gene
# Source: Myles
polyp_target = pd.read_csv(TARGET_DIR / "adenoma_risk_age60.csv", index_col="gene")

# Polyp cumulative risk to age 70 --------------------------------------------
# By: sex, age
# Source: Mecklin
polyp_cumulative_target = pd.read_csv(
    TARGET_DIR / "adenoma_risk_cumulative_to_70.csv", index_col=["sex", "age"]
)

# Other ----------------------------------------------------------------------
# Checkpoint CRC, adenoma, adv adenoma dev at first colo (no prev)
# By: gene
# Source: Governe 2020
first_colo_target = pd.read_csv(
    TARGET_DIR / "crc_aden_first_colo_checkpoint.csv", index_col=["gene"]
)
adn_dwell_target = pd.read_csv(TARGET_DIR / "adn_dwell_time.csv", index_col=["gene"])


# ---------------------------------------------------------------------------- #
# SCREENING TEST SPECS
# ---------------------------------------------------------------------------- #

screening_specs_df = pd.read_csv(
    DATA_DIR / "screening_test_specs.csv", index_col=["test", "variable"]
)
# Only need estimate -- not using upper and lower bounds currently
SCREENING_TEST_SPECS = screening_specs_df.loc[:, "estimate"]
