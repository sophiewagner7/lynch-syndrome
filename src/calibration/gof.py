import numpy as np

import configs as c
import configs.inputs as i


# Calculate score based on difference between model outputs and targets
def objective(log):
    inc, _, inc_log, _ = log
    score = 0

    # Yearly incidence penalty (ages 20-84)
    score += np.square(
        inc_log[4, :65].cumsum() - 100000 * i.incidence_target.loc[:65, "value"]
    ).sum()

    # # Polyp prevalence penalty (pooled)
    # score += (1 / np.sqrt(35656)) * np.square(inc_log[12, :65].sum() - c.N * c.polyp_targets[1])

    return None
