import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

# --- Load & clean data ---
df_raw = pd.read_pickle("data/df_merged_v3.pickle")

# Map IDs to pseudonyms
json_file_path = "config/id_to_pseudonym.json"
# Load the mapping into a Python dictionary
with open(json_file_path, "r", encoding="utf-8") as f:
    id_to_pseudonym = json.load(f)

df_raw["pseudonym"] = df_raw["patient_id"].map(id_to_pseudonym)
df_raw = df_raw.dropna(subset=["pseudonym"])

# --- Restrict data to first 365 days per participant ---
df_raw["timestamp_utc"] = pd.to_datetime(
    df_raw["timestamp_utc"]
)  # ensure datetime format

# Sort values to get correct chronological order per participant
df_raw = df_raw.sort_values(by=["pseudonym", "timestamp_utc"])

# Calculate the first timestamp per participant
first_dates = df_raw.groupby("pseudonym")["timestamp_utc"].transform("min")

# Compute days since first timestamp
df_raw["days_since_start"] = (df_raw["timestamp_utc"] - first_dates).dt.days

# Keep only rows within first 365 days
df_raw = df_raw[df_raw["days_since_start"] <= 364]

df_raw = df_raw.drop(
    columns=["timestamp_utc", "woche_PHQ9_sum", "patient_id", "days_since_start"]
)

# --- Group by participant ---
grouped = df_raw.groupby("pseudonym")

# --- Summary stats across participants ---
stats_data = []

for col in df_raw.columns:
    if col == "pseudonym":
        continue

    participant_num_records = grouped[col].count()
    participant_means = grouped[col].mean()

    num_records_mean = participant_num_records.mean()
    num_records_std = participant_num_records.std()

    values_mean_mean = participant_means.mean()
    values_mean_std = participant_means.std()

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

stats_df = pd.DataFrame(stats_data)

# Save feature-level summary
stats_df.to_csv("results/df_raw_column_stats.csv", index=False)
print("Saved column stats to results/df_raw_column_stats.csv")

# Display in notebook
markdown_table = stats_df.to_markdown(index=False)
display(Markdown(markdown_table))

# --- Compute per-participant feature statistics + missing ratio ---
# Core stats
participant_stats = grouped.agg(["mean", "median", "max", "min", "std"])

# Flatten multi-index columns
participant_stats.columns = [f"{col}_{stat}" for col, stat in participant_stats.columns]

# Compute missing values per participant and feature
missing_counts = grouped.apply(lambda g: g.isna().sum())
total_counts = grouped.size().rename("total_samples")  # total rows per participant

# Expand total_counts to match shape of missing_counts
total_counts_expanded = pd.DataFrame(
    np.repeat(total_counts.values[:, np.newaxis], missing_counts.shape[1], axis=1),
    index=missing_counts.index,
    columns=missing_counts.columns,
)

missing_ratio = (missing_counts / total_counts_expanded) * 100
missing_ratio.columns = [f"{col}_missing_ratio" for col in missing_ratio.columns]

# Combine everything, including total sample count
participant_stats_full = pd.concat([participant_stats, missing_ratio], axis=1)
participant_stats_full["total_samples"] = total_counts

# Save to CSV
participant_stats_full.to_csv("results/per_participant_feature_stats.csv")
print(
    "Saved per-participant stats with missing ratios to results/per_participant_feature_stats.csv"
)

# --- Plot variation of abend_PHQ2_sum across participants with improved annotations ---

# Compute counts, missing, total, and missing rate
counts = df_raw.groupby("pseudonym")["abend_PHQ2_sum"].count()
missing = df_raw.groupby("pseudonym")["abend_PHQ2_sum"].apply(lambda x: x.isna().sum())
total = counts + missing
missing_rate = (missing / total * 100).round(1)

# Create sorted list of pseudonyms for consistent plotting
sorted_pseudonyms = counts.index.tolist()

# Set up the figure
plt.figure(figsize=(12, 6))
ax = sns.boxplot(
    data=df_raw, x="pseudonym", y="abend_PHQ2_sum", order=sorted_pseudonyms
)

# Annotate with stats above each box
y_max = df_raw["abend_PHQ2_sum"].max()
for i, pseudonym in enumerate(sorted_pseudonyms):
    phq2_count = counts.get(pseudonym, 0)
    miss_pct = missing_rate.get(pseudonym, 0)
    total_samples = total.get(pseudonym, 0)

    annotation = (
        f"#phq2: {phq2_count}\n" f"#tot: {total_samples}\n" f"mis: {miss_pct:.1f}%"
    )

    ax.text(
        i,
        y_max + 1.5,  # Raised annotation higher
        annotation,
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=0,
    )

# Finalize plot
plt.xticks(rotation=90)
# plt.title("Variation of abend_PHQ2_sum across participants")
plt.xlabel("Participant")
plt.ylabel("abend_PHQ2_sum")
plt.tight_layout()
plt.savefig("results/abend_PHQ2_sum_variation.png", dpi=300)
plt.show()
print("✅ Saved plot: results/abend_PHQ2_sum_variation.png")
