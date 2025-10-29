import pandas as pd

target_first_colo = {
    "MLH1": {
        "p_aden": 0.2727,
        "se_aden": 0.0601,
        "p_adv": 0.1455,
        "se_adv": 0.0475,
        "p_crc": 0.0545,
        "se_crc": 0.0306,
        "n": 55,
        "med_age": 38,
    },
    "MSH2": {
        "p_aden": 0.2273,
        "se_aden": 0.0633,
        "p_adv": 0.0909,
        "se_adv": 0.0433,
        "p_crc": 0.0227,
        "se_crc": 0.0225,
        "n": 44,
        "med_age": 39,
    },
    "MSH6": {
        "p_aden": 0.2587,
        "se_aden": 0.0366,
        "p_adv": 0.1189,
        "se_adv": 0.0271,
        "p_crc": 0.0280,
        "se_crc": 0.0138,
        "n": 143,
        "med_age": 49,
    },
    "PMS2": {
        "p_aden": 0.3636,
        "se_aden": 0.0272,
        "p_adv": 0.1818,
        "se_adv": 0.0822,
        "p_crc": 0.0140,
        "se_crc": 0.0444,
        "n": 22,
        "med_age": 50,
    },
}

target_polyp_dwell_time = {
    "MLH1": {
        "med_yrs": 4,
        "iqr_lower_yrs": 2,
        "iqr_upper_yrs": 8,
        "range_lower_yrs": 0,
        "range_upper_yrs": 24,
        "n": 36,
        "med_age": 49,
        "iqr_lower_age": 41,
        "iqr_upper_age": 60,
    },
    "MSH2": {
        "med_yrs": 3,
        "iqr_lower_yrs": 1,
        "iqr_upper_yrs": 6,
        "range_lower_yrs": 0,
        "range_upper_yrs": 17,
        "n": 42,
        "med_age": 49,
        "iqr_lower_age": 46,
        "iqr_upper_age": 67,
    },
    "MSH6": {
        "med_yrs": 3,
        "iqr_lower_yrs": 2,
        "iqr_upper_yrs": 6,
        "range_lower_yrs": 0,
        "range_upper_yrs": 14,
        "n": 97,
        "med_age": 49,
        "iqr_lower_age": 53,
        "iqr_upper_age": 68,
    },
    "PMS2": {
        "med_yrs": 1,
        "iqr_lower_yrs": 1,
        "iqr_upper_yrs": 4,
        "range_lower_yrs": 0,
        "range_upper_yrs": 6,
        "n": 16,
        "med_age": 49,
        "iqr_lower_age": 38,
        "iqr_upper_age": 59,
    },
}

import pandas as pd

# Convert nested dicts to DataFrames
df_first_colo = (
    pd.DataFrame(target_first_colo).T.reset_index().rename(columns={"index": "gene"})
)
df_polyp_dwell = (
    pd.DataFrame(target_polyp_dwell_time)
    .T.reset_index()
    .rename(columns={"index": "gene"})
)

# Save to CSV
df_first_colo.to_csv("target_first_colo.csv", index=False)
df_polyp_dwell.to_csv("target_polyp_dwell_time.csv", index=False)

print("Saved:")
print(" - target_first_colo.csv")
print(" - target_polyp_dwell_time.csv")
