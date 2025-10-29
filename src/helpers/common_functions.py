import numpy as np
import configs.global_configs as c


def probtoprob(prob, a=1, b=12):
    """
    Convert probability over period a to probability over period b.
    Default is converting annual probability to monthly probability.
    """
    prob = np.clip(prob, 0, 1 - 1e-10)  # ensure valid probability
    return 1 - (1 - prob) ** (a / b)


def probtohaz(prob, a=1, b=12):
    """
    Convert probability over period a to hazard over period b.
    Default is converting annual probability to monthly hazard.
    """
    prob = np.clip(prob, 0, 1 - 1e-10)  # avoid log(0)
    return -np.log1p(-prob) * (a / b)


def haztoprob(hazard, a=1, b=12):
    """
    Convert hazard over period a to probability over period b.
    Default: convert annual hazard to monthly probability.
    """
    hazard = np.clip(hazard, 0.0, np.inf)
    return 1.0 - np.exp(-hazard * (b / a))


# ---------------- helpers ----------------


def age_to_idx(age):
    """Age in years -> index on the model age axis (yearly)."""
    return int(age - c.START_AGE)


def isfinite(x):
    return (x is not None) and np.isfinite(x)


def weight(var=None, n=None):
    """Prefer 1/var if finite & >0, else 1/sqrt(n) if finite & >0, else 1."""
    if isfinite(var) and var > 0:  # type: ignore
        return 1.0 / float(var)  # type: ignore
    if isfinite(n) and n > 0:  # type: ignore
        return 1.0 / float(np.sqrt(n))  # type: ignore
    return 1.0


def w_hinge(y, lo=None, hi=None, *, var=None, n=None):
    """
    Zero loss if y in [lo, hi] (when both bounds present).
    If only one bound present, behave like one-sided hinge.
    Outside the band, use squared loss.
    Weighted by 1/var, else 1/sqrt(n), else 1.
    """
    # Resolve missing bounds to act as one-sided constraints
    lo_ok = isfinite(lo)
    hi_ok = isfinite(hi)
    if not lo_ok and not hi_ok:
        # no band -> pure 0 loss
        return 0.0

    if lo_ok and y < lo:
        r = lo - y
    elif hi_ok and y > hi:
        r = y - hi
    else:
        r = 0.0

    return weight(var, n) * (r * r)


def _check_dims(a):
    if a.ndim not in (3, 4):
        raise ValueError("inc_unadj must be (sex,state,age) or (sex,gene,state,age)")


def _axis_map(a):
    """Return dict mapping semantic names -> axis indices for a's shape."""
    if a.ndim == 3:
        return {"sex": 0, "gene": 1, "age": 2}
    return {"sex": 0, "age": 1}


def _select_state(a, state_idx, *, age_stop=None) -> np.ndarray:
    """
    Select one state and optional age slice.
    Returns an array with trailing dimension = age.
    Shapes:
      4D -> (sex, gene, age) after selection
      3D -> (sex, age) after selection
    """
    _check_dims(a)

    if a.ndim == 4:
        out = a[:, :, state_idx, :]
    else:
        out = a[:, state_idx, :]

    if age_stop is not None:
        out = out[..., :age_stop]  # keep 'age' as last axis
    return out


def _sum_over(a, keys):
    """
    Sum over the listed semantic axes if present in array `a`.
    Keys can include any of {"sex","gene","state","age"} but typically
    you'll use {"sex","gene","age"} after selecting a single state.
    """
    if not keys:
        return a
    ax = _axis_map(a)
    axes = tuple(sorted({ax[k] for k in keys if k in ax}))
    return a.sum(axis=axes)


def _denominator(sum_over) -> float:
    """
    Build the proportion denominator based on what you summed over.
    By default: POP_SIZE x (#sexes if 'sex' in sum_over) x (#genes if 'gene' in sum_over).
    Set include_genes=False if you DON'T want to scale by n_genes.
    """
    denom = float(c.POPULATION_SIZE)
    if sum_over is not None:
        if "sex" in sum_over:
            denom *= float(c.n_sexes)
        if "gene" in sum_over:
            denom *= float(c.n_genes)
    return denom


# ----------


def cumulative_cases_to_age(
    inc_unadj,
    age_idx,
    state_idx=0,
    sum_over=("sex", "age"),
    return_prop=True,
) -> np.ndarray:
    """
    Sum incident cases up to age_idx along requested axes.
    Works for (sex,gene,state,age) or (sex,state,age).
    If return_prop=True, divides by POP_SIZE x (#sexes if summed) x (#genes if summed)
    """
    _check_dims(inc_unadj)
    arr = _select_state(inc_unadj, state_idx, age_stop=age_idx)  # (..., age)
    total = _sum_over(arr, sum_over)

    if not return_prop:
        return total

    denom = _denominator(sum_over)
    return total / denom


def cumsum_cases(
    inc_unadj,
    state_idx=0,
    sum_over=None,
    return_prop=True,
) -> np.ndarray:
    """
    Cumulative incidence (proportion) over age.
    - 4D input -> returns (sex, gene, age)
    - 3D input -> returns (sex, age)
    You can optionally sum some axes first (e.g., sum_over=("sex",) to pool sexes).
    """
    _check_dims(inc_unadj)

    arr = _select_state(inc_unadj, state_idx)  # select state, keep age as last axis
    if sum_over is not None:
        arr = _sum_over(arr, sum_over)  # optional pre-sum (e.g., over sex)
    cum = np.cumsum(arr, axis=-1)  # cumulative over age (last axis)

    if not return_prop:
        return cum

    denom = _denominator(sum_over)
    return cum / denom
