import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import helpers.common_functions as func
import configs.global_configs as c


# Function to extract transition probabilities
def extract_transition_probs(tmat, states, transitions):
    transition_probs = {}
    for from_state, to_state in transitions:
        from_idx = states[from_state]
        to_idx = states[to_state]
        params = tmat[:, from_idx, to_idx]
        transition_probs[f"{from_state} to {to_state}"] = params
    return transition_probs
