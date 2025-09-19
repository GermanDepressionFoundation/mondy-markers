import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from utils import PLOT_STYLES, add_logo_to_figure

plt.rcParams.update({"font.family": PLOT_STYLES["font"]})

RESULT_DIR = "results2"

# --- Load feature importance stats ---
elastic_csv = f"{RESULT_DIR}/elasticnet_feature_importance_stats.csv"
rf_csv = f"{RESULT_DIR}/randomforest_feature_importance_stats.csv"

elastic_df = pd.read_csv(elastic_csv, index_col=0)
rf_df = pd.read_csv(rf_csv, index_col=0)

# --- Merge into one dataframe ---
merged = pd.DataFrame(
    {"ElasticNet_raw": elastic_df["mean"], "RF_raw": rf_df["mean"]}
).fillna(0)

# Normalize each by its max (0–1 scale)
merged["ElasticNet"] = merged["ElasticNet_raw"] / merged["ElasticNet_raw"].max()
merged["RF"] = merged["RF_raw"] / merged["RF_raw"].max()

# Compute shift after normalization
merged["shift"] = merged["RF"] - merged["ElasticNet"]
merged["abs_shift"] = merged["shift"].abs()

# Categorize shift
merged["change_type"] = np.where(
    merged["shift"] > 0,
    "↑ More important in RF",
    np.where(merged["shift"] < 0, "↓ More important in EN", "No change"),
)

# --- Highlight top x shifted features for annotations ---
TOP_HIGHLIGHT = 20
top_shift_features = merged.sort_values("abs_shift", ascending=False).head(
    TOP_HIGHLIGHT
)

# # --- Scatter Plot ---
# fig = px.scatter(
#     merged.reset_index().rename(columns={"index": "feature"}),
#     x="ElasticNet",
#     y="RF",
#     color="change_type",
#     template="plotly_white",
#     size="abs_shift",
#     hover_name="feature",
#     hover_data={
#         "ElasticNet": ":.3f",
#         "RF": ":.3f",
#         "shift": ":.3f",
#         "abs_shift": ":.3f",
#         "change_type": True,
#     },
#     color_discrete_map={
#         "↑ More important in RF": PLOT_STYLES["colors"]["RF"],
#         "↓ More important in EN": PLOT_STYLES["colors"]["Elasticnet"],
#         "No change": "gray",
#     },
#     title="Normalized Feature Importance Shift: Elastic Net vs Random Forest",
#     labels={
#         "ElasticNet": "Elastic Net (Mean Feature Importance, Normalized 0–1)",
#         "RF": "Random Forest (Mean Feature Importance, Normalized 0–1)",
#     },
# )

# # Add diagonal line y=x
# fig.add_shape(
#     type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", dash="dash")
# )

# fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))

# fig.update_layout(
#     width=1200,
#     height=1200,
#     legend_title="Shift Type",
# )

# # Add vertical annotations (90° rotated, BELOW with margin)
# for feat, row in top_shift_features.iterrows():
#     fig.add_annotation(
#         x=row["ElasticNet"],
#         y=row["RF"],
#         text=feat,
#         textangle=90,
#         showarrow=False,
#         font=dict(size=10, color="black"),
#         xanchor="center",
#         yanchor="top",
#         yshift=-12,
#     )

# # Add logo to Plotly figure
# fig.add_layout_image(
#     dict(
#         source="../SDD_Logo_rgb_pos_600_reduced2.png",
#         xref="paper",
#         yref="paper",
#         x=0.1,
#         y=0.9,
#         sizex=0.2,
#         sizey=0.2,
#         xanchor="left",
#         yanchor="top",
#         layer="above",
#     )
# )

# # --- Paths ---
# png_path = f"{RESULT_DIR}/feature_importance_shift_scatter.png"
# html_path = f"{RESULT_DIR}/feature_importance_shift_scatter.html"

# # Save PNG
# fig.write_image(png_path, scale=1)

# # Create descriptive text
# description_text = f"""
# <h2>What this scatter plot shows</h2>
# <p>
# Each point represents a single feature used in the models. The x-axis shows its <b>normalized importance in Elastic Net</b>
# (a linear model), while the y-axis shows its <b>normalized importance in Random Forest</b>
# (a non-linear model).<br><br>

# - Points along the <b>diagonal dashed line</b> have similar importance in both models.<br>
# - <span style="color:red;"><b>Red points</b></span> gained importance in the non-linear model (RF).<br>
# - <span style="color:orange;"><b>Orange points</b></span> were more important in the linear model (EN).<br>
# - Point size reflects how strongly the importance shifted between models.<br><br>

# Hover over any point to see the feature name and exact values.<br>
# The <b>top {TOP_HIGHLIGHT} most shifted features</b> are annotated directly on the plot.
# </p>
# """

# # Embed figure HTML inside a simple 2-column layout
# fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

# full_html = f"""
# <html>
# <head>
# <title>Feature Importance Shift</title>
# <style>
# body {{
#     font-family: Arial, sans-serif;
#     margin: 20px;
# }}
# .container {{
#     display: flex;
#     flex-direction: row;
# }}
# .text-panel {{
#     width: 30%;
#     padding-right: 20px;
# }}
# .plot-panel {{
#     width: 70%;
# }}
# </style>
# </head>
# <body>
# <div class="container">
#     <div class="text-panel">
#         {description_text}
#     </div>
#     <div class="plot-panel">
#         {fig_html}
#     </div>
# </div>
# </body>
# </html>
# """

# # Save combined HTML
# with open(html_path, "w", encoding="utf-8") as f:
#     f.write(full_html)

# print(f"Saved PNG: {png_path}")
# print(f"Saved interactive HTML with description: {html_path}")

# --- Aggregated Feature Importance (Bar Plots) ---

# Load feature importance counts
elasticnet_features = pd.read_csv(
    f"{RESULT_DIR}/elasticnet_top_features.csv", header=None, index_col=0
).squeeze("columns")
rf_features = pd.read_csv(
    f"{RESULT_DIR}/randomforest_top_features.csv", header=None, index_col=0
).squeeze("columns")

# Combine feature counts into a DataFrame and fill missing values
bar_width = 0.4
feature_df = pd.DataFrame(
    {"ElasticNet": elasticnet_features, "RandomForest": rf_features}
).fillna(0)

# Ensure consistent order
feature_df = feature_df.sort_values(["ElasticNet", "RandomForest"], ascending=False)
x = range(len(feature_df))

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    [i - bar_width / 2 for i in x],
    feature_df["ElasticNet"],
    width=bar_width,
    label="Elastic Net",
    color=PLOT_STYLES["colors"]["Elasticnet"],
)
ax.bar(
    [i + bar_width / 2 for i in x],
    feature_df["RandomForest"],
    width=bar_width,
    label="Random Forest",
    color=PLOT_STYLES["colors"]["RF"],
)
ax.set_xticks(list(x))
ax.set_xticklabels(feature_df.index, rotation=90)
ax.set_ylabel("Number of Appearances in Top-K")
plt.title("Top 20 Features: Elastic Net vs Random Forest")
plt.legend()

# Add logo
add_logo_to_figure(fig, position="top_right")

plt.tight_layout()
plt.savefig(
    f"{RESULT_DIR}/aggregated_feature_importance_comparison_grouped.png",
    dpi=300,
)
plt.close()

# --- Merge mean values ---
feature_means = pd.DataFrame(
    {"ElasticNet_raw": elastic_df["mean"], "RandomForest_raw": rf_df["mean"]}
).fillna(0)

# Normalize by each model's max
feature_means["ElasticNet"] = (
    feature_means["ElasticNet_raw"] / feature_means["ElasticNet_raw"].max()
)
feature_means["RandomForest"] = (
    feature_means["RandomForest_raw"] / feature_means["RandomForest_raw"].max()
)

# Sort by Elastic Net relative importance
feature_means_sorted = feature_means.sort_values("ElasticNet", ascending=False)

# Pick top N for readability
TOP_N = len(feature_means)
top_feature_means = feature_means_sorted.head(TOP_N)

x = range(len(top_feature_means))
bar_width = 0.4

fig, ax = plt.subplots(figsize=(14, 6))

# Elastic Net bars (left)
ax.bar(
    [i - bar_width / 2 for i in x],
    top_feature_means["ElasticNet"],
    width=bar_width,
    label="Elastic Net (relative)",
    color=PLOT_STYLES["colors"]["Elasticnet"],
)

# Random Forest bars (right)
ax.bar(
    [i + bar_width / 2 for i in x],
    top_feature_means["RandomForest"],
    width=bar_width,
    label="Random Forest (relative)",
    color=PLOT_STYLES["colors"]["RF"],
)

ax.set_xticks(list(x))
ax.set_xticklabels(top_feature_means.index, rotation=90)
ax.set_ylabel("Mean Feature Importance (Normalized 0–1)")
plt.title(f"Mean Feature Importance (Normalized, Sorted by Elastic Net)")
plt.legend()

# Add logo
add_logo_to_figure(fig, position="top_right")

plt.tight_layout()
plt.savefig(
    f"{RESULT_DIR}/aggregated_feature_importance_relative_sorted_elastic.png", dpi=300
)
plt.close()

# --- Normalize each model's importance by its max ---
elastic_df["relative"] = elastic_df["mean"] / elastic_df["mean"].max()
rf_df["relative"] = rf_df["mean"] / rf_df["mean"].max()

# --- Sort by each model's relative importance ---
elastic_sorted = elastic_df.sort_values("relative", ascending=False)
rf_sorted = rf_df.sort_values("relative", ascending=False)

# --- Select top N for readability ---
elastic_top = elastic_sorted.head(TOP_N)
rf_top = rf_sorted.head(TOP_N)

# --- Plot Elastic Net ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    elastic_top.index,
    elastic_top["relative"],
    color=PLOT_STYLES["colors"]["Elasticnet"],
    label="Elastic Net (relative importance)",
)
plt.xticks(rotation=90)
plt.ylabel("Relative Mean Feature Importance (0–1)")
plt.title(f"Features (Elastic Net, normalized by max)")

# Add logo
add_logo_to_figure(fig, position="top_right")

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/elasticnet_relative_feature_importance.png", dpi=300)
plt.close()

# --- Plot Random Forest ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    rf_top.index,
    rf_top["relative"],
    color=PLOT_STYLES["colors"]["RF"],
    label="Random Forest (relative importance)",
)
plt.xticks(rotation=90)
plt.ylabel("Relative Mean Feature Importance (0–1)")
plt.title(f"Features (Random Forest, normalized by max)")

# Add logo
add_logo_to_figure(fig, position="top_right")

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/randomforest_relative_feature_importance.png", dpi=300)
plt.close()
