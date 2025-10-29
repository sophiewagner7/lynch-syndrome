import pandas as pd
import numpy as np

from .markov import MarkovModel
from configs import c, inputs
from helpers import common_functions as func

# MarkovModel: D[t,s] gives us distribution at time t
# MarkovModel: I[t,s] gives us incidence at time t


cancer_costs_init = [54491, 73135, 88769, 115554]
cancer_costs_cont = [4017, 3780, 5296, 16041]
cancer_costs_term = [84050, 83716, 83307, 116339]

cancer_qalys_init = [54491, 73135, 88769, 115554]
cancer_qalys_cont = [4017, 3780, 5296, 16041]
cancer_qalys_term = [84050, 83716, 83307, 116339]

csd_1 = func.probtoprob(0.05)
csd_2 = func.probtoprob(0.19)
csd_3 = func.probtoprob(0.75)
csd_4 = func.probtoprob(0.90)
csds = [csd_1, csd_2, csd_3, csd_4]


# We need to take cancers once detected, and apply age-specific death rates, also to apply costs
def run_cancer_microsim(model: MarkovModel, seed=42) -> pd.DataFrame:

    newly_diagnosed = model.I[:, 7:11]
    acm = inputs.lt_1y[f"prob_death_{model.spec.sex}_m"]
    out_rows = []
    id = 0
    for t in range(c.NUM_CYCLES):
        age_months = c.START_AGE + t
        diagnosed_at_t = newly_diagnosed[t]

        for stage in [0, 1, 2, 3]:

            for _ in range(int(diagnosed_at_t[stage])):

                age_dx = age_months // 12
                cycles_survived = 0
                cause = "alive_end"
                age_death = np.nan
                cycle_death = np.nan
                p_csd = csds[stage]

                while True:
                    age = age_months // 12
                    p_acm = float(acm[age])

                    if cycles_survived >= 60:
                        p_crc = 0.0

                    # competing risks: total death prob this cycle
                    # convert to hazards to avoid sum>1 issues
                    h_acm = -np.log(1.0 - min(0.999999, p_acm)) if p_acm > 0 else 0.0
                    h_csd = -np.log(1.0 - min(0.999999, p_csd)) if p_csd > 0 else 0.0
                    h_tot = h_acm + h_csd
                    p_any = 1.0 - np.exp(-h_tot) if h_tot > 0 else 0.0

                    u = np.random.rand()
                    if u < p_any:
                        # death occurs; choose cause proportionally to hazards
                        prob_csd_given_death = (h_csd / h_tot) if h_tot > 0 else 0.0
                        cause = (
                            "csd"
                            if (np.random.rand() < prob_csd_given_death)
                            else "acm"
                        )
                        age_death = age
                        cycle_death = t
                        break

                    # else survive cycle
                    age_months += 1
                    cycles_survived += 1

                    # guard: don't run forever
                    if cycles_survived >= 60:
                        break

                # buckets to make costs/QALYs dead simple later
                first_year_months = int(min(12, cycles_survived))
                continuing_months = int(max(0, cycles_survived - 24))
                terminal_flag = 1 if cause in ("csd", "acm") else 0

                out_rows.append(
                    {
                        "id": int(id),
                        "gene": model.spec.gene,
                        "sex": model.spec.sex,
                        "stage": stage + 1,
                        "age_dx": int(age_dx),
                        "cycle_dx": int(t),
                        "cause_of_death": cause,  # 'crc' | 'bg' | 'alive_end'
                        "cycles_survived": int(cycles_survived),
                        "cycle_death": (
                            int(cycle_death) if not np.isnan(cycle_death) else np.nan
                        ),
                        "age_death": (
                            float(age_death) if not np.isnan(age_death) else np.nan
                        ),
                        "first_year_months": int(
                            first_year_months
                        ),  # apply first-year costs/QALYs to these months (<=12)
                        "continuing_months": int(
                            continuing_months
                        ),  # apply continuing costs/QALYs to these months
                        "terminal_flag": int(
                            terminal_flag
                        ),  # terminal bucket present (1) or not (0)
                    }
                )
                id += 1

    return pd.DataFrame(out_rows)


def summarize_microsim_for_costs(per_person: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one-row totals scaled by weights:
      - cases_simulated, cases_weight_total
      - total_first_year_months, total_continuing_months, total_terminal_flags
      - crc_deaths, bg_deaths, alive_end
      - stage mix (weighted)
      - avg_years_survived_by_stage (weighted)
    """

    one = np.ones(len(per_person))
    out = {
        "cases_simulated": int(len(per_person)),
        "total_first_year_months": float((per_person["first_year_months"]).sum()),
        "total_continuing_months": float((per_person["continuing_months"]).sum()),
        "total_terminal_flags": float((per_person["terminal_flag"]).sum()),
        "csd_deaths": float((one * (per_person["cause_of_death"] == "csd")).sum()),
        "acm_deaths": float((one * (per_person["cause_of_death"] == "acm")).sum()),
        "alive_end": float((one * (per_person["cause_of_death"] == "alive_end")).sum()),
    }

    # stage mix and mean years survived by stage
    for s in (1, 2, 3, 4):
        mask = per_person["stage"] == s
        out[f"stage{s}_cases_w"] = float(one * mask.sum())
        if one[mask].sum() > 0:
            out[f"stage{s}_avg_years_survived"] = float(
                (per_person.loc[mask, "years_survived"]).sum() / one[mask].sum()
            )
        else:
            out[f"stage{s}_avg_years_survived"] = 0.0

    return pd.DataFrame([out])
