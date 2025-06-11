import numpy as np

POPULATION_SIZE = 100_000
CYCLE_LENGTH = 1 / 12  # units = months
NUM_YEARS = 50  # duration of the model
NUM_CYCLES = NUM_YEARS / CYCLE_LENGTH  # years
START_AGE = 25
END_AGE = START_AGE + int(NUM_YEARS)

time = range(int(NUM_CYCLES))  # Stages
age_time = range(START_AGE, END_AGE)  # Ages

D_RATE = 0.03  # Discount rate
GENES = ["MLH1", "MSH2", "MSH6", "PMS2"]  # Gene variants (cohorts)
AGES = [25, 30, 35, 40]  # Starting surveillance ages
INTERVALS = [1, 2, 3, 4, 5]  # Number of years between surveillance
SEXES = ["male", "female", "both"]  # Sex
ADH = 0.6  # Adherence rate
WTP = 100000  # Willingness to pay threshold

AGE_LAYERS = np.arange(
    START_AGE, END_AGE + 1, 5
)  # Age layers for transition probabilities

# ---------------------------------------------------------------- #
# MARKOV MODEL SETTINGS
# ---------------------------------------------------------------- #
all_states_itos = {
    0: "healthy",  # healthy state
    1: "lr_polyp",  # low-risk polyp
    2: "hr_polyp",  # high-risk polyp
    3: "u_stage_1",  # undiagnosed CRC stage I
    4: "u_stage_2",  # undiagnosed CRC stage II
    5: "u_stage_3",  # undiagnosed CRC stage III
    6: "u_stage_4",  # undiagnosed CRC stage IV
    7: "d_stage_1",  # diagnosed CRC stage I
    8: "d_stage_2",  # diagnosed CRC stage II
    9: "d_stage_3",  # diagnosed CRC stage III
    10: "d_stage_4",  # diagnosed CRC stage IV
    11: "death_cancer",  # death from cancer
    12: "death_all_cause",  # all cause death
    13: "death_colo",  # death from colonoscopy screening
}
all_states_stoi = {v: k for k, v in all_states_itos.items()}  # State mapping

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
n_states = len(all_states_itos.keys())  # Total number of states


# State Connectivity -------------------------

transitions_idx = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (4, 8),
    (5, 9),
    (6, 10),
]

transitions_str = [
    ("healthy", "LR_polyp"),
    ("LR_polyp", "HR_polyp"),
    ("HR_polyp", "u_CRC_loc"),
    ("u_CRC_loc", "u_CRC_reg"),
    ("u_CRC_reg", "u_CRC_dis"),
    ("u_CRC_loc", "d_CRC_loc"),
    ("u_CRC_reg", "d_CRC_reg"),
    ("u_CRC_dis", "d_CRC_dis"),
]

transitions_itos = zip(transitions_idx, transitions_str)
transitions_stoi = zip(transitions_str, transitions_idx)

# TRANSITION PROBABILITIES
tmat = {
    gene: {sex: f"data/tmats/{gene}_{sex}_tmat.npy"} for gene in GENES for sex in SEXES
}

init_tp_values = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
init_tps = {transitions_idx[i]: init_tp_values[i] for i in range(len(transitions_idx))}


acm_rate = {0: "male", 1: "female"}
