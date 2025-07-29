import numpy as np
import pandas as pd
from IPython.display import Markdown, display

df_raw = pd.read_pickle("data/df_merged_v3.pickle")
df_raw = df_raw.drop(columns=["timestamp_utc", "woche_PHQ9_sum"])

# Group by participant
grouped = df_raw.groupby("patient_id")

stats_data = []

# Iterate over each column except patient_id itself
for col in df_raw.columns:
    if col == "patient_id":
        continue

    # Per participant: number of records & mean values
    participant_num_records = grouped[col].count()  # number of non-NaN per participant
    participant_means = grouped[col].mean()  # mean value per participant

    # Across participants
    num_records_mean = participant_num_records.mean()
    num_records_std = participant_num_records.std()

    values_mean_mean = participant_means.mean()  # mean of participant means
    values_mean_std = participant_means.std()  # std across participant means

    # Missing rate = total missing values / total rows for this column
    total_missing = df_raw[col].isna().sum()
    missing_rate = (total_missing / len(df_raw)) * 100

    stats_data.append(
        {
            "Feature": col,
            "NumRecords_mean": num_records_mean,
            "NumRecords_std": num_records_std,
            "Values_mean": values_mean_mean,
            "Values_std": values_mean_std,
            "MissingRate (%)": missing_rate,
        }
    )

# Create stats dataframe
stats_df = pd.DataFrame(stats_data)

# Save to CSV
stats_df.to_csv("results/df_raw_column_stats.csv", index=False)
print("Saved column stats to results/df_raw_column_stats.csv")

# Create Markdown table for Jupyter
markdown_table = stats_df.to_markdown(index=False)
display(Markdown(markdown_table))
