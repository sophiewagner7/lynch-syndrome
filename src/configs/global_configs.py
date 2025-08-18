import numpy as np

# ---------------------------------------------------------------------------- #
# GLOBAL PARAMETERS
# ---------------------------------------------------------------------------- #

POPULATION_SIZE = 100_000
START_AGE = 20
END_AGE = 75
CYCLE_LENGTH = 1 / 12  # 1 month
NUM_YEARS = END_AGE - START_AGE  # years
NUM_CYCLES = int(NUM_YEARS / CYCLE_LENGTH)  # cycles

time = range(NUM_CYCLES)  # Cycles
age_time = range(START_AGE, END_AGE)  # Ages

D_RATE = 0.03  # Annual discount rate
WTP = 100_000  # Willingness-to-pay threshold (per QALY)
ADH = 0.6  # Adherence rate

# ---------------------------------------------------------------------------- #
# COHORT SETTINGS
# ---------------------------------------------------------------------------- #

GENES = ["MLH1", "MSH2", "MSH6", "PMS2"]
SEXES = ["male", "female"]
SCREENING_START_AGES = [25, 30, 35, 40]  # Surveillance start ages
SCREENING_INTERVALS = [1, 2, 3, 4, 5]  # Surveillance intervals (years)
ALT_TESTS = ["FIT", "sDNA"]  # Non-colo tests for interdigitation

# Map age bands to indices (5-year increments)
AGE_LAYERS = {age: idx for idx, age in enumerate(np.arange(START_AGE, END_AGE, 5))}

# ---------------------------------------------------------------------------- #
# HEALTH STATES
# ---------------------------------------------------------------------------- #

health_states_itos = {
    0: "healthy",
    1: "lr_polyp",
    2: "hr_polyp",
    3: "u_stage_1",
    4: "u_stage_2",
    5: "u_stage_3",
    6: "u_stage_4",
    7: "d_stage_1",
    8: "d_stage_2",
    9: "d_stage_3",
    10: "d_stage_4",
    11: "death_cancer",
    12: "death_all_cause",
    13: "death_colo",
}
health_states_stoi = {v: k for k, v in health_states_itos.items()}

alive_states = [
    "healthy",
    "lr_polyp",
    "hr_polyp",
    "u_stage_1",
    "u_stage_2",
    "u_stage_3",
    "u_stage_4",
    "d_stage_1",
    "d_stage_2",
    "d_stage_3",
    "d_stage_4",
]
death_states = ["death_cancer", "death_all_cause", "death_colo"]

# ---------------------------------------------------------------------------- #
# TRANSITIONS TO CALIBRATE
# ---------------------------------------------------------------------------- #

calibration_tps_itos = {
    (0, 1): ("healthy", "LR_polyp"),
    (1, 2): ("LR_polyp", "HR_polyp"),
    (2, 3): ("HR_polyp", "u_stage_1"),
    (3, 4): ("u_stage_1", "u_stage_2"),
    (4, 5): ("u_stage_2", "u_stage_3"),
    (5, 6): ("u_stage_3", "u_stage_4"),
    (3, 7): ("u_stage_1", "d_stage_1"),
    (4, 8): ("u_stage_2", "d_stage_2"),
    (5, 9): ("u_stage_3", "d_stage_3"),
    (6, 10): ("u_stage_4", "d_stage_4"),
}
calibration_tps_stoi = {v: k for k, v in calibration_tps_itos.items()}

# ---------------------------------------------------------------------------- #
# REFERENCE COUNTS
# ---------------------------------------------------------------------------- #

n_states = len(health_states_itos)
n_genes = len(GENES)
n_sexes = len(SEXES)
n_age_layers = len(AGE_LAYERS)
n_calibration_tps = len(calibration_tps_itos)
