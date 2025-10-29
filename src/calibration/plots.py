import sys
from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import helpers.common_functions as func
from configs import c, inputs


# Function to extract transition probabilities
def extract_transition_probs(tmat, states, transitions):
    transition_probs = {}
    for from_state, to_state in transitions:
        from_idx = states[from_state]
        to_idx = states[to_state]
        params = tmat[:, from_idx, to_idx]
        transition_probs[f"{from_state} to {to_state}"] = params
    return transition_probs


GENES_ALL = ["MLH1", "MSH2", "MSH6", "PMS2"]
GENE2IDX = {g: i for i, g in enumerate(GENES_ALL)}


def plot_crc_incidence(
    log: np.ndarray,
    gene: str,
    state_idx=[
        0,
    ],
) -> None:

    func._check_dims(log)  # (S,G,A,N) or (S,A,N)
    ages = np.arange(20, 85, 1)

    # Handle both single state idx and multiple (if multiple detected cancer states)
    IDX_UCRC = 5
    if state_idx is None:
        k_idx = np.array([IDX_UCRC], dtype=int)
    else:
        k_idx = np.atleast_1d(state_idx).astype(int)
    model_cuminc = np.zeros_like(log)
    model_cuminc = func._select_state(model_cuminc, 0)  # collapse along correct axis
    for k in k_idx:
        model_cuminc += func.cumsum_cases(log, k)

    # Result: (S,G,A) or (S,A)

    # Select desired gene
    if model_cuminc.ndim == 3 and model_cuminc.shape[1] == 4:
        model_cuminc = model_cuminc[:, GENE2IDX[gene], :].copy()
    elif model_cuminc.ndim == 3 and model_cuminc.shape[1] == 1:
        model_cuminc = model_cuminc[:, 0, :].copy()  # single gene
    # Else, passed (S, A, N) inc_unadj, meaning already have single gene
    # Result: (S, A) selection

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for s, sex in enumerate(c.SEXES):

        ax = axes[s]

        inc = inputs.incidence_cumulative_target.loc[
            (gene, sex), ["value", "lower", "upper"]
        ]
        yerr_lower = inc["value"] - inc["lower"]
        yerr_upper = inc["upper"] - inc["value"]
        yerr = [yerr_lower, yerr_upper]

        ax.scatter(inc.index, inc["value"])
        ax.errorbar(
            x=inc.index,
            y=inc["value"],
            yerr=yerr,
            fmt="o",  # marker style
            color="black",
            ecolor="gray",  # error bar color
            capsize=3,  # small horizontal cap lines
            elinewidth=1,
            capthick=1,
            label="Target (Dominguez 2023)",
        )

        ax.plot(ages, model_cuminc[s], label="Model", color="tab:blue")

        # --- aesthetics ---
        ax.set_title(f"{sex.capitalize()} {gene}")
        ax.set_xlabel("Age")
        if s == 0:
            ax.set_ylabel("Cumulative CRC incidence (%)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{gene}: Model vs Target Cumulative CRC Incidence", fontsize=14, y=1.02
    )
    fig.tight_layout()
    plt.show()


def plot_polyp_incidence(log: np.ndarray, gene: str, state_idx: int) -> None:
    """Plot polyp incidence
    Args:
        log (np.ndarray): incidence log, (S,G,A,N) or (S,A,N)
        gene (str): options ["MLH1","MSH2","MSH6","PMS2"]
        state_idx (int): polyp state (LR or HR)
    """
    func._check_dims(log)  # (S,G,A,N) or (S,A,N)
    ages = np.arange(20, 85, 1)
    model_cuminc = func.cumsum_cases(log, state_idx)

    # Select desired gene
    if model_cuminc.ndim == 3 and model_cuminc.shape[1] == 4:
        model_cuminc = model_cuminc[:, GENE2IDX[gene], :].copy()
    elif model_cuminc.ndim == 3 and model_cuminc.shape[1] == 1:
        model_cuminc = model_cuminc[:, 0, :].copy()  # single gene

    # Else, passed (S, A, N) inc_unadj, meaning already have single gene
    # Result: (S, A) selection

    # Single target value for both sexes
    target_val = inputs.polyp_target.loc[gene]

    # Still, plot sexes separately
    colors = {"female": "red", "male": "blue"}
    for s, sex in enumerate(c.SEXES):
        plt.plot(
            ages,
            model_cuminc[s],
            color=colors[sex],
            label=f"{sex.capitalize()} {gene} polyp incidence",
        )
    plt.scatter([60], [target_val], color="black", label="Target at age 60")
    plt.title(f"{gene} Polyp Incidence")
    plt.legend()
    plt.show()


def plot_transition(tmat: np.ndarray, transition: tuple[int, int]) -> None:
    """Pass tmat (S,A,N,N) with hazards and plot
    Args:
        tmat (np.ndarray): (S,A,N,N) with hazards
        transtion (tuple[int,int]): transition idxs, e.g. (0,3)
    """
    from_state, to_state = transition

    colors = {"female": "red", "male": "blue"}
    for s, sex in enumerate(c.SEXES):
        plt.plot(
            list(c.AGE_LAYERS.keys()),
            tmat[s, :, from_state, to_state],
            color=colors[sex],
            label=f"{sex.capitalize()}",
        )

    plt.title(f"Transition from {c.calibration_tps_itos[transition]}")
    plt.legend()
    plt.show()


def _mean_geometric_dwell_years(p_stay_monthly):
    """
    Expected time in a discrete-time state with per-step 'stay' probability p_ii.
    E[T_months] = 1 / (1 - p_ii), converted to years by dividing by 12.
    """
    p = np.clip(p_stay_monthly, 0.0, 1.0 - 1e-12)
    dwell_months = 1.0 / (1.0 - p)
    return dwell_months / 12.0


def calc_total_polyp_dwell_time(tmat, lr_idx, hr_idx):
    """
    Compute mean total dwell time (years) in LR + HR polyp states,
    averaged over age layers, for each sex.

    Parameters
    ----------
    tmat : ndarray, shape (S, A, N, N)
        Monthly transition matrix by sex (S), age layer (A), and states (N x N).
    lr_idx, hr_idx : int
        Indices for LR and HR polyp states.

    Returns
    -------
    dwell_df : pd.DataFrame with mean dwell times (years) per sex.
    """
    eps = 1e-9

    # Convert hazards to probabilities per month (outgoing)
    p_lr = func.haztoprob(tmat[:, :, lr_idx, :], 1, 1)  # (S, A, N)
    p_hr = func.haztoprob(tmat[:, :, hr_idx, :], 1, 1)  # (S, A, N)

    # Per-month probability of *staying* in the same state
    # p_stay = 1 - sum(outgoing probs excluding self)
    p_stay_lr = 1.0 - (p_lr.sum(axis=2) - p_lr[:, :, lr_idx])
    p_stay_hr = 1.0 - (p_hr.sum(axis=2) - p_hr[:, :, hr_idx])

    p_stay_lr = np.clip(p_stay_lr, 0.0, 1.0 - eps)
    p_stay_hr = np.clip(p_stay_hr, 0.0, 1.0 - eps)

    # Convert to dwell time (years) and average over age layers
    dwell_lr = _mean_geometric_dwell_years(p_stay_lr).mean(axis=1)  # mean over age
    dwell_hr = _mean_geometric_dwell_years(p_stay_hr).mean(axis=1)

    # Sum LR + HR dwell times for each sex
    dwell_total = dwell_lr + dwell_hr

    return pd.DataFrame({"sex": c.SEXES, "mean_dwell_years": dwell_total})


def calc_preclinical_sojourn_time(tmat, state_idx=None):
    """
    Compute mean preclinical sojourn time (years) in uCRC states,
    averaged over age layers, for each sex.

    Parameters
    ----------
    tmat : ndarray, shape (S, A, N, N)
        Monthly transition matrix by sex (S), age layer (A), and states (N x N).
    state_idx : arr[int]
        Index (or indices) for uCRC state(s).

    Returns
    -------
    dwell_df : pd.DataFrame with mean dwell times (years) per sex.
    """
    eps = 1e-9

    # Defualt
    IDX_UCRC = 5
    if state_idx is None:
        k_idx = np.array([IDX_UCRC], dtype=int)
    else:
        k_idx = np.atleast_1d(state_idx).astype(int)

    # Convert hazards -> monthly transition probabilities: shape (S, A, N, N)
    P = func.haztoprob(tmat)

    # For each state k: p_stay_k = 1 - sum_j P[k->j] + P[k->k]
    dwell_per_k = []
    for k in k_idx:
        Pk = P[:, :, k, :]  # (S, A, N)
        stay = 1.0 - Pk.sum(axis=2) + Pk[:, :, k]  # (S, A)
        stay = np.clip(stay, 0.0, 1.0 - eps)
        # Convert to expected dwell (years), then average over age layers -> (S,)
        dwell_years = _mean_geometric_dwell_years(stay).mean(axis=1)
        dwell_per_k.append(dwell_years)

    dwell_per_k = np.stack(dwell_per_k, axis=0)  # (K, S)
    dwell_total = dwell_per_k.sum(axis=0)  # (S,)

    return pd.DataFrame({"sex": list(c.SEXES), "mean_dwell_years": dwell_total})


def get_stage_dist(result_log):
    inc_unadj = result_log[2]
    stage_idx = [7, 8, 9, 10]  # stage_1..stage_4
    stage_totals = inc_unadj[:, stage_idx, :].sum(axis=(2)).sum(axis=0)
    total_crc = stage_totals.sum()

    # Proportion in each stage:
    stage_props = stage_totals / total_crc
    stage_labels = ["stage1", "stage2", "stage3", "stage4"]
    stage_totals_df = pd.DataFrame(stage_props.reshape(1, -1), columns=stage_labels)
    return stage_totals_df


def get_total_incidence(log, state_idx):
    inc_unadj = log[2]
    func._check_dims(inc_unadj)

    # Defualt
    IDX_UCRC = 5
    if state_idx is None:
        k_idx = np.array([IDX_UCRC], dtype=int)
    else:
        k_idx = np.atleast_1d(state_idx).astype(int)

    age_idx = func.age_to_idx(80)

    total_inc_per_sex = np.zeros_like(inc_unadj)
    total_inc_per_sex = func._select_state(total_inc_per_sex, 0)
    for state in state_idx:
        total_inc_per_sex += func.cumulative_cases_to_age(
            inc_unadj, age_idx, state, sum_over=("age",)
        )

    return total_inc_per_sex
