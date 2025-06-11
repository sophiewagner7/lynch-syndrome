import pathlib as pl
from enum import Enum

import pandas as pd
import numpy as np


# -----------------------
# Enums
# -----------------------

class StrategyType(Enum):
    NH = ("nh", "No screening")
    CURRENT = ("cur", "Current guideline (annual at 25)")
    EXPERIMENTAL = ("exp", "New test strategy")

    def __init__(self, code, description):
        self.code = code
        self.description = description

class RunMode(Enum):
    """
    Enum for the different run modes of the model.
    """
    NH = "nh"
    CURRENT = "cur"
    EXPERIMENTAL = "exp"
    CALIBRATION = "calibration"
    ICER = "icer"
    PSA = "psa"

# -----------------------
# Global Settings
# -----------------------

CYCLE_LENGTH = 1  # units = years
NUM_YEARS = 50  # duration of the model
NUM_CYCLES = NUM_YEARS / CYCLE_LENGTH  # years
START_AGE = 25
END_AGE = START_AGE + int(NUM_YEARS)

time = range(int(NUM_CYCLES))  # Stages
age_time = range(START_AGE, END_AGE)  #  Ages

D_RATE = 0.03  # Discount rate
GENES = ["MLH1", "MSH2", "MSH6", "PMS2"]  # Gene variants (cohorts)
AGES = [25, 30, 35, 40]  # Starting surveillance ages
INTERVALS = [1, 2, 3, 4, 5]  # Number of years between surveillance
GENDERS = ["male", "female", "both"]  # Sex
ADH = 0.6  # Adherence rate
WTP = 100000 # Willingness to pay threshold

# -----------------------
# Directory Structure
# -----------------------

SRC_DIR = pl.Path.cwd().parent
DATA_DIR = SRC_DIR / "Data"
DUMP_DIR = SRC_DIR / "Dump"
RESULTS_DIR = DUMP_DIR / "Results"
GRAPHS_DIR = RESULTS_DIR / "Graphs"

# Folders 
OS_RESULTS_DIR = GRAPHS_DIR / "OS_Results"
CRC_RESULTS_DIR = GRAPHS_DIR / "CRC_Results"
CD_RESULTS_DIR = GRAPHS_DIR / "CD_Results"
OWSA_DIR = GRAPHS_DIR / "OWSA"
EFFICIENCY_FRONTIERS_DIR = GRAPHS_DIR / "Efficiency_Frontiers"
PSA_DIR = GRAPHS_DIR / "PSA"
D_MATRICES_DIR = RESULTS_DIR / "D_matrices"
TABLES_DIR = RESULTS_DIR / "Tables"
ICER_TABLE_PATH = TABLES_DIR / "ICERS"
OWSA_TABLE_PATH = TABLES_DIR / "OWSA"
MISC_DIR = SRC_DIR.parent / "Misc"

# -----------------------
# File Paths
# -----------------------

# TODO: Confirm if males/females modeled separately or together
params_male = DATA_DIR / "model_inputs.xlsx"
params_female = DATA_DIR / "model_inputs.xlsx"

# TODO: Identify what this risk is for
nh_risk_dict = {
    "MLH1": DATA_DIR / "Nono_MLH1.csv",
    "MLH2": DATA_DIR / "Nono_MSH2.csv",
    "MSH6": DATA_DIR / "Nono_MSH6.csv",
    "PMS2": DATA_DIR / "Nono_PMS2.csv",
}

# Gene specific adenoma and advanced adenoma risks
risk_adenoma_dict = {
    "MLH1": DATA_DIR / "MLH1_Adenoma_Risk.csv",
    "MSH2": DATA_DIR / "MSH2_Adenoma_Risk.csv",
    "MSH6": DATA_DIR / "MSH6_Adenoma_Risk.csv",
    "PMS2": DATA_DIR / "PMS2_Adenoma_Risk.csv",
}
adv_risk_adenoma_dict = {
    "MLH1": DATA_DIR / "MLH1_Lynch_Adv_Adenoma_Engel.csv",
    "MSH2": DATA_DIR / "MSH2_Lynch_Adv_Adenoma_Engel.csv",
    "MSH6": DATA_DIR / "MSH6_Lynch_Adv_Adenoma_Engel.csv",
    "PMS2": DATA_DIR / "MSH6_Lynch_Adv_Adenoma_Engel.csv",
}

# TODO: Figure out how this data was generated and how used
# 10 year survival curves
srvl_colon = DATA_DIR / "Survival_colon.csv"
srvl_rectal = DATA_DIR / "Survival_rectal.csv"
srvl_crc = DATA_DIR / "Survival_CRC.npy"

# -----------------------
# Strategy presets
# -----------------------

all_optimal_strats = [[5, 40], [4, 40], [3, 40], [3, 35], [3, 25], [2, 25], [1, 25]]

# TODO: What is BK?
BC_BK_STRATS = {
    "MLH1": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH2": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH6": [[5, 40], [3, 40], [3, 35], [1, 25]],
    "PMS2": [[5, 40], [4, 40], [3, 40], [3, 35], [1, 25]],
}
BC_FLOOR_STRATS = {
    "MLH1": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH2": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH6": [[3, 40], [3, 35], [1, 25]],
    "PMS2": [[4, 40], [3, 40], [1, 25]],
}
BC_OPTIM_STRATS = {
    "MLH1": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH2": [[3, 35], [3, 25], [2, 25], [1, 25]],
    "MSH6": [[3, 40], [3, 35]],
    "PMS2": [[4, 40], [3, 40]],
}
NAT_HIST = {"MLH1": [[0, 25]], "MSH2": [[0, 25]], "MSH6": [[0, 25]], "PMS2": [[0, 25]]}

# -----------------------
# States and Transitions
# -----------------------

# to get stage, multiply cancer_prob * stage_dist[run_spec.interval]
ALL_STATES = {
    0: "mutation",  # step 0 in all models
    1: "cur",  # healthy state for current guideline
    2: "exp",  # healthy state for experimental
    3: "nh",  # healthy state for nh
    4: "init adenoma",  # initial adenoma found
    5: "adenoma",  # previously had an adenoma on last csy
    6: "init dx stage I",  # first year of having CRC stage I
    7: "init dx stage II",  # first year of having CRC stage II
    8: "init dx stage III",  # first year of having CRC stage III
    9: "init dx stage IV",  # first year of having CRC stage IV
    10: "dx stage I",  # CRC stage I diagnosis
    11: "dx stage II",  # CRC stage II diagnosis
    12: "dx stage III",  # CRC stage III diagnosis
    13: "dx stage IV",  # CRC stage IV diagnosis
    14: "all cause dx",  # all cause death while having cancer
    15: "all cause",  # all cause daeth while in healthy state
    16: "cancer death",  # death from cancer
    17: "csy death",  # death from colonoscopy screening
    18: "stage I death",  # last year of life having CRC stage I
    19: "stage II death",  # last year of lif having CRC stage II
    20: "stage III death",  # last year of life having CRC stage III
    21: "stage IV death",  # last year of life having CRC stage IV
}

# connects between states in the model
CONNECTIVITY = {
    0: [1, 2, 3],  # chooses which model (current, new, nh)
    1: [1, 4, 6, 7, 8, 9, 15, 17],
    2: [2, 4, 6, 7, 8, 9, 15, 17],
    3: [3, 6, 7, 8, 9, 15],
    4: [5, 6, 7, 8, 9, 15, 17],
    5: [5, 6, 7, 8, 9, 15, 17],
    6: [10, 14, 18],
    7: [11, 14, 19],
    8: [12, 14, 20],
    9: [13, 14, 21],
    10: [10, 14, 16],
    11: [11, 14, 16],
    12: [12, 14, 16],
    13: [13, 14, 16],
    14: [14],
    15: [15],
    16: [16],
    17: [17],
    18: [16],
    19: [16],
    20: [16],
    21: [16]
}

# -----------------------
# Run Specification
# -----------------------

# class that gives the parameters for a certain run
class RunSpec:
    """
    Class to hold the parameters for a specific run of the model.
    """
    def __init__(self, interval: int, gene: str, gender: str, start_age: int=25):
        self.interval = interval
        self.gene = gene
        self.gender = gender
        self.start_age = start_age
        self.risk_ratio = risk_ratios[str(interval)]

        self.strategy = self.determine_strategy()
        self.interval_str = f"Q{self.interval}dY"
        self.label = f", CSY Q{self.interval}Y, Start Age: {self.start_age}"
        self.file_name = self.gene + self.label

    def determine_strategy(self)-> str:
        if self.interval == 0: 
            return StrategyType.NH
        elif self.interval == 1:
            return StrategyType.CURRENT
        else: 
            return StrategyType.EXPERIMENTAL


# -----------------------
# Parameters and Data
# -----------------------

# TODO: clarify sources and input
# make 1-3 the same
# icers with life-years
# 1-3 calculated as average of engel et al and 4-5 with jarvinen et al CI bounds
# this combination seemed to produce the best results
# These values are intended to manually code risk reduction from screening
risk_ratios = {
    "0": "NONO",
    "1": 0.215 * 1.02,  # .215
    "2": 0.274,  # .274
    "3": 0.304,
    "4": 0.6,
    "5": 0.829,
}

# From Visser et al. Corresponds to <60 years, <70 years, <80 years
colectomy_death_risk = [0.03, 0.08, 0.06]

# Baseline risk?
risk_adn_male_data = DATA_DIR / "Adenoma_Risk.csv" # only this is used
risk_adn_female_data = DATA_DIR / "female_adenoma_risk.csv" # not found in code

# Adenoma risk ratios (click et al)
adv_adenoma_prob = 0.3625
adv_adenoma_risk_mult = 2

# Complication rates from colonoscopy (levin et al 2008)
p_csy_comp_no_bx = 0.8 / 1000.0
p_csy_comp_bx = 7 / 1000.0
p_csy_death = 1 / 16318.0

# Standardized Mortality Rate (dejong et al)
# Applied to ACM to get the lynch specific ACM
male_SMR_Lynch = 1.0
female_SMR_Lynch = 1.0

# ? What does this do?
csy_risk_reduc = 0.60
csy_srvl_reduc = 0.80

# Distribution of CRC Staging
# comes from male but variables are the same for each gender
# I believe these are for calibration
staging = pd.read_excel(params_male, "Stage Dists", index_col=0)

# ? Come back to this. How is this by stage?
# CRC death rates by stage
# to get CRC death: death_rate[stage] - this_gender.lynch_ac_mortality[age] if result is + else lynch_ac_mort
CRC_death_rate = pd.read_excel(params_male, "CRC Survival", index_col=0)

# Misc. Probabilities
# More detailed complication rates, like perforation, bleed
misc = pd.read_excel(params_male, "Misc", index_col=0)

# Costs for the model
costs = pd.read_excel(params_male, "Costs", index_col=0)

# Clean csy = without biopsy
# dx_csy = with biopsy (slightly greater cost)
clean_csy = costs.iloc[0, 0]
dx_csy = costs.iloc[1, 0]

# Utilities
UTILS = pd.read_excel(params_male, "Utilities", index_col=0)
# these are the same sheets with different names
UTILS_F = pd.read_csv(DATA_DIR / "full_util_table_f.csv") # not used
UTILS_M = pd.read_csv(DATA_DIR / "full_util_table_f.csv") # only used, why female?
# only used in lynch_icer.py, not in running of markov.

# Disutilities
csy_disutil = UTILS.iloc[8, 0]  # dw Colonoscopy
compl_disutil = UTILS.iloc[9, 0]  # dw Colo complication


# extracting probabilites from life tables for male
LT = pd.ExcelFile(DATA_DIR / "Weighted_lifetable.xlsx").parse("Sheet1")
LT_prob_male = LT.ix[:, 2]
LT_rate_male = -(np.log(1 - LT_prob_male))
Lynch_LT_rate_male = LT_rate_male * male_SMR_Lynch
Lynch_LT_prob_male = 1 - np.exp(-abs(Lynch_LT_rate_male) * CYCLE_LENGTH)

# use ac mortality data beginning at starting age:
startIndex = int(START_AGE / CYCLE_LENGTH)
lynch_ac_mortality = Lynch_LT_prob_male[startIndex:]
cut_ac_mortality = LT_prob_male[startIndex:]

all_cause_OS = DATA_DIR / "all_cause_survival.npy"

LT_prob_female = LT.ix[:, 1]
LT_rate_female = -(np.log(1 - LT_prob_female))
Lynch_LT_rate_female = LT_rate_female * female_SMR_Lynch
Lynch_LT_prob_female = 1 - np.exp(-abs(Lynch_LT_rate_female) * CYCLE_LENGTH)

# use ac mortality data beginning at starting age:
lynch_ac_mortality_fm = Lynch_LT_prob_female[startIndex:]
cut_ac_mortality_fm = LT_prob_female[startIndex:]

cut_ac_mortality_cmb = (cut_ac_mortality + cut_ac_mortality_fm) / 2.0

# -----------------------
# Strategy list
# -----------------------
# TODO: Confirm if this is used in the model

strat_list = np.load("strat_list.npy")

MLH1_psa_strat = ["Q2Y, Start age: 25", "Q1Y, Start age: 25"]
MSH2_psa_strat = ["Q3Y, Start age: 25", "Q2Y, Start age: 25"]
MSH6_psa_strat = ["Q4Y, Start age: 40", "Q3Y, Start age: 40"]
PMS2_psa_strat = ["Q4Y, Start age: 40", "Q3Y, Start age: 40"]

STRAT_LIST_PATH = DATA_DIR / "strat_list.npy"

def make_strat_list():
    strat_list = [
        f"{gene} Q{intrvl}Y, Start age: {age}"
        for gene in GENES for intrvl in INTERVALS for age in AGES
    ]
    np.save(STRAT_LIST_PATH, np.array(strat_list))
    return strat_list
