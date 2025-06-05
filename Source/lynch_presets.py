# Lynch presets

"""

Description: Presets to run the LS markov model
Authors: Myles Ingram, Elisabeth Silver
Last update: 07.02.19

"""

import pathlib as pl
import pandas as pd
import numpy as np

# Global Settings
CYCLE_LENGTH = 1  # units = years
NUM_YEARS = 50  # duration of the model
NUM_CYCLES = NUM_YEARS / CYCLE_LENGTH  # years
START_AGE = 25
END_AGE = START_AGE + int(NUM_YEARS)

time = range(int(NUM_CYCLES))  # Stages
age_time = range(START_AGE, END_AGE)  #  Ages
dRate = 0.03  # Discount rate
genes = ["MLH1", "MSH2", "MSH6", "PMS2"]  # Gene variants (cohorts)
csy_protocols = ["nono", "current", "new"]  # Surveillance strat options
ages = [25, 30, 35, 40]  # Starting surveillance ages
intervals = [1, 2, 3, 4, 5]  # Number of years between surveillance
genders = ["male", "female", "both"]  # Sex
adh = 0.6  # Adherence rate

# Define current vs nono vs new (these are horrible names)
ALL_RUN_MODES = [
    "CURRENT MARKOV",
    "NONO MARKOV",
    "NEW MARKOV",
    "CALIBRATION",
    "ICER",
    "PSA",
]

all_optimal_strats = [[5, 40], [4, 40], [3, 40], [3, 35], [3, 25], [2, 25], [1, 25]]

# What is BK?
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

# to get stage, multiply cancer_prob * stage_dist[run_spec.interval]
ALL_STATES = {
    0: "mutation",  # step 0 in all models
    1: "current",  # healthy state for current
    2: "new",  # healthy state for new
    3: "nono",  # healthy state for nono
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
    #              22: 'init adv adenoma', # initial adv adenoma found
    #              23: 'adv adenoma' # previously had advance adenoma on last csy
}


# connects between states in the model
CONNECTIVITY = {
    0: [1, 2, 3],  # chooses which model (current, new, nono)
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
    21: [16],
    #                22: [6, 7, 8, 9, 15, 17, 23],
    #                23: [6, 7, 8, 9, 15, 17, 23]
}


# class that gives the parameters for a certain run
class run_type:

    def __init__(self, interval, gene, gender_spec, age_spec=25):
        self.risk_ratio = risk_ratios[str(interval)]
        self.interval = interval
        self.gene = gene
        self.gender = gender_spec
        self.start_age = age_spec

        if interval == 1:
            self.guidelines = "current"
        elif interval == 0:
            self.guidelines = "nono"
            self.start_age = 25
        else:
            self.guidelines = "new"
        self.interval_str = "Q%dY" % (self.interval)
        self.label = ", CSY Q%dY, Start Age: %d" % (self.interval, self.start_age)
        self.file_name = self.gene + self.label


# Folders
src = pl.Path.cwd().parent
data_repo = src / "Data"
dump = src / "Dump"
results = dump / "Results"
graphs = results / "Graphs"
os_results = graphs / "OS_Results"
crc_results = graphs / "CRC_Results"
cd_results = graphs / "CD_Results"
owsa = graphs / "OWSA"
ef = graphs / "Efficiency_Frontiers"
psa = graphs / "PSA"
D_matrices = results / "D_matrices"
tables = results / "Tables"
icer_table = tables / "ICERS"
owsa_table = tables / "OWSA"
misc = src.parent / "Misc"

# Probs for model

table = pd.ExcelFile(data_repo / "model_inputs.xlsx")
# model_inputs_male = table.parse("Model Inputs Male")
# model_inputs_female = table.parse("Model Inputs Female")

# ?Are males and females modeled separately?
params_male = data_repo / "model_inputs.xlsx"
params_female = data_repo / "model_inputs.xlsx"

# Natural history cumulative risk?
nono_mlh1 = data_repo / "Nono_MLH1.csv"
nono_msh2 = data_repo / "Nono_MSH2.csv"
nono_msh6 = data_repo / "Nono_MSH6.csv"
nono_pms2 = data_repo / "Nono_PMS2.csv"


NONO_RISK = {
    "MLH1": nono_mlh1,
    "MSH2": nono_msh2,
    "MSH6": nono_msh6,
    "PMS2": nono_pms2,
}

# 10 year survival curves
# ?How was this data generated? How are these values applied?
srvl_colon = data_repo / "Survival_colon.csv"
srvl_rectal = data_repo / "Survival_rectal.csv"
srvl_crc = data_repo / "Survival_CRC.npy"


# From Moller et al
# =============================================================================
# risk_ratios = { "0": "NONO",
#                 "1": 0.06,
#                 "2": 0.31,
#                 "3": 0.65,
#                 "4": 0.79,
#                 "5": 0.92,
#         }
# =============================================================================
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

# =============================================================================
# risk_ratios = { "0": "NONO",
#                 "1": 0.377,
#                 "2": 0.377,
#                 "3": 0.377,
#                 "4": 0.6,
#                 "5": 0.829,
#         }
#
#
# =============================================================================

# =============================================================================
# #from jarvinen et al confidence intervals (assuming center = Q3)
# risk_ratios = { "0": "NONO",
#                 "1": 0.171,
#                 "2": 0.274,
#                 "3": 0.377,
#                 "4": 0.60299,
#                 "5": 0.829,
#         }
#
# risk_range_ratios = { "1-2": .48,
#                       "2-3": 1,
#                       "3-4": 1.22,
#                       "4-5": 1.41,
#                       "5+": 1.54
#         }
# =============================================================================

WTP = 100000

# From Visser et al. Corresponds to <60 years, <70 years, <80 years
colectomy_death_risk = [0.03, 0.08, 0.06]

# Gene specific adenoma and advanced adenoma risks
risk_adenoma_dict = {
    "MLH1": data_repo / "MLH1_Adenoma_Risk.csv",
    "MSH2": data_repo / "MSH2_Adenoma_Risk.csv",
    "MSH6": data_repo / "MSH6_Adenoma_Risk.csv",
    "PMS2": data_repo / "PMS2_Adenoma_Risk.csv",
}
adv_risk_adenoma_dict = {
    "MLH1": data_repo / "MLH1_Lynch_Adv_Adenoma_Engel.csv",
    "MSH2": data_repo / "MSH2_Lynch_Adv_Adenoma_Engel.csv",
    "MSH6": data_repo / "MSH6_Lynch_Adv_Adenoma_Engel.csv",
    "PMS2": data_repo / "MSH6_Lynch_Adv_Adenoma_Engel.csv",
}

# Baseline risk?
risk_adenoma = 0.1
risk_adn_male_data = data_repo / "Adenoma_Risk.csv"
risk_adn_female_data = data_repo / "female_adenoma_risk.csv"

# from click et al
adv_adenoma_prob = 0.3625
adv_adenoma_risk_mult = 2

# from levin et al 2008
p_csy_comp_no_bx = 0.8 / 1000.0
p_csy_comp_bx = 7 / 1000.0
p_csy_death = 1 / 16318.0

# from dejong et al
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
util = pd.read_excel(params_male, "Utilities", index_col=0)
# these are the same sheets with different names
utils_f = pd.read_csv(data_repo / "full_util_table_f.csv")
utils_m = pd.read_csv(data_repo / "full_util_table_f.csv")

# Disutilities
csy_disutil = util.iloc[8, 0]  # dw Colonoscopy
compl_disutil = util.iloc[9, 0]  # dw Colo complication


# extracting probabilites from life tables for male
xl = pd.ExcelFile(data_repo / "Weighted_lifetable.xlsx")
lifeTable = xl.parse("Sheet1")
LT_prob_male = lifeTable.ix[:, 2]
LT_rate_male = -(np.log(1 - LT_prob_male))
Lynch_LT_rate_male = LT_rate_male * male_SMR_Lynch
Lynch_LT_prob_male = 1 - np.exp(-abs(Lynch_LT_rate_male) * CYCLE_LENGTH)

# use ac mortality data beginning at starting age:
startIndex = int(START_AGE / CYCLE_LENGTH)
lynch_ac_mortality = Lynch_LT_prob_male[startIndex:]
cut_ac_mortality = LT_prob_male[startIndex:]

all_cause_OS = data_repo / "all_cause_survival.npy"

LT_prob_female = lifeTable.ix[:, 1]
LT_rate_female = -(np.log(1 - LT_prob_female))
Lynch_LT_rate_female = LT_rate_female * female_SMR_Lynch
Lynch_LT_prob_female = 1 - np.exp(-abs(Lynch_LT_rate_female) * CYCLE_LENGTH)

# use ac mortality data beginning at starting age:
lynch_ac_mortality_fm = Lynch_LT_prob_female[startIndex:]
cut_ac_mortality_fm = LT_prob_female[startIndex:]

cut_ac_mortality_cmb = (cut_ac_mortality + cut_ac_mortality_fm) / 2.0

strat_list = np.load("strat_list.npy")

MLH1_psa_strat = ["Q2Y, Start age: 25", "Q1Y, Start age: 25"]
MSH2_psa_strat = ["Q3Y, Start age: 25", "Q2Y, Start age: 25"]
MSH6_psa_strat = ["Q4Y, Start age: 40", "Q3Y, Start age: 40"]
PMS2_psa_strat = ["Q4Y, Start age: 40", "Q3Y, Start age: 40"]


def make_strat_list():
    Q5_list = []
    for gene in genes:
        for intrvl in intervals:
            for age in ages:
                strat = gene + " Q" + str(intrvl) + "Y, Start age: " + str(age)
                Q5_list.append(strat)
    strat_list = np.array(Q5_list)
    #    print(strat_list)
    np.save("strat_list.npy", strat_list)
    return


# output_dict = {'D_matrix_MLH1_1_25_both': pd.read_csv('../current_D_matrix/anneal_test_MLH1_D_matrix'),
#               'D_matrix_MSH2_1_25_both': pd.read_csv('../current_D_matrix/anneal_test_MSH2_D_matrix'),
#               'D_matrix_MSH6_1_25_both': pd.read_csv('../current_D_matrix/anneal_test_MSH6_D_matrix'),
#               'D_matrix_PMS2_1_25_both': pd.read_csv('../current_D_matrix/anneal_test_PMS2_D_matrix')
#        }
