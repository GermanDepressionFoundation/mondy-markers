import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import PLOT_STYLES, add_logo_to_figure

plt.rcParams["font.family"] = PLOT_STYLES["font"]

RESULTS_DIR = "results"
results_df = pd.read_csv(f"{RESULTS_DIR}/model_performance_summary.csv", index_col=0)
results_df = results_df.sort_index()
# Sort by RF R² for consistency
# result_df = results_df.sort_values("r2_rf")

# Combined R² Plot
fig = plt.figure(figsize=(12, 5))
bar_width = 0.35
indices = range(len(results_df))

plt.bar(
    indices,
    results_df["r2_elastic"],
    width=bar_width,
    label="Elastic Net R²",
    color=PLOT_STYLES["colors"]["Elasticnet"],
)
plt.bar(
    [i + bar_width for i in indices],
    results_df["r2_rf"],
    width=bar_width,
    label="Random Forest R²",
    color=PLOT_STYLES["colors"]["RF"],
)

plt.ylim(-1, 1.0)
plt.axhline(0.3, color="red", linestyle="--", label="R² = 0.3 Threshold")
plt.xticks([i + bar_width / 2 for i in indices], results_df.index, rotation=90)
plt.ylabel("R²")
plt.title("R² Comparison Across Participants")
plt.legend()
plt.tight_layout()
add_logo_to_figure(fig)
plt.savefig(f"{RESULTS_DIR}/plot_r2_comparison_per_participant.png", dpi=300)
plt.close()

# Combined MAE Plot
fig = plt.figure(figsize=(12, 5))
plt.bar(
    indices,
    results_df["mae_elastic"],
    width=bar_width,
    label="Elastic Net MAE",
    color=PLOT_STYLES["colors"]["Elasticnet"],
)
plt.bar(
    [i + bar_width for i in indices],
    results_df["mae_rf"],
    width=bar_width,
    label="Random Forest MAE",
    color=PLOT_STYLES["colors"]["RF"],
)

plt.ylim(0, 5.0)
plt.axhline(2.0, color="red", linestyle="--", label="MAE = 2.0 Threshold")
plt.xticks([i + bar_width / 2 for i in indices], results_df.index, rotation=90)
plt.ylabel("MAE")
plt.title("MAE Comparison Across Participants")
plt.legend()
plt.tight_layout()
add_logo_to_figure(fig)
plt.savefig(f"{RESULTS_DIR}/plot_mae_comparison_per_participant.png", dpi=300)
plt.close()


# Define responder criteria
responder_rf = (results_df["r2_rf"] > 0.3) | (results_df["mae_rf"] < 1.0)
responder_elastic = (results_df["r2_elastic"] > 0.3) | (results_df["mae_elastic"] < 1.0)

results_df["responder_rf"] = responder_rf
results_df["responder_elastic"] = responder_elastic

# Count responders
responder_counts = pd.DataFrame(
    {
        "Responder": [responder_rf.sum(), responder_elastic.sum()],
        "Non-Responder": [(~responder_rf).sum(), (~responder_elastic).sum()],
    },
    index=["Random Forest", "Elastic Net"],
)

# Save responder table
responder_counts.to_csv(f"{RESULTS_DIR}/responder_counts.csv")

# --- Model Winner Summary ---


def model_winner(row):
    rf_better = row["r2_rf"] > row["r2_elastic"] and row["mae_rf"] < row["mae_elastic"]
    elastic_better = (
        row["r2_elastic"] > row["r2_rf"] and row["mae_elastic"] < row["mae_rf"]
    )
    if rf_better:
        return "Random Forest"
    elif elastic_better:
        return "Elastic Net"
    else:
        return "Tie or Mixed"


results_df["model_winner"] = results_df.apply(model_winner, axis=1)
model_winner_summary = results_df["model_winner"].value_counts()
model_winner_summary.to_csv(f"{RESULTS_DIR}/model_winner_summary.csv")

# Extract RMSSD values
rmssd_phq2_values = results_df["rmssd_phq2"].values

# Compute 25th percentile thresholds
phq2_thresh = np.nanpercentile(rmssd_phq2_values, 25)

print(f"PHQ-2 25th percentile threshold: {phq2_thresh:.3f}")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 1, figsize=(12, 6), sharey=False)

# --- Left: PHQ-2 RMSSD ---
axes.boxplot(
    rmssd_phq2_values,
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="skyblue", color="black"),
    medianprops=dict(color="black"),
)
axes.axvline(
    phq2_thresh,
    color="red",
    linestyle="--",
    label=f"25th percentile = {phq2_thresh:.3f}",
)
axes.set_title("PHQ-2 RMSSD")
axes.set_xlabel("RMSSD")
axes.legend(loc="upper right")

plt.suptitle(
    "Within-Person Symptom Variability (RMSSD) with 25th Percentile Thresholds"
)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/phq_rmssd_thresholds.png", dpi=300)
