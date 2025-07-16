import numpy as np
import pandas as pd
import plotly.express as px

# --- Load feature importance stats ---
elastic_csv = "results/elasticnet_feature_importance_stats.csv"
rf_csv = "results/randomforest_feature_importance_stats.csv"

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

# --- Highlight top 20 shifted features for annotations ---
TOP_HIGHLIGHT = 20
top_shift_features = merged.sort_values("abs_shift", ascending=False).head(
    TOP_HIGHLIGHT
)

# --- Scatter Plot ---
fig = px.scatter(
    merged.reset_index().rename(columns={"index": "feature"}),
    x="ElasticNet",
    y="RF",
    color="change_type",
    size="abs_shift",
    hover_name="feature",
    hover_data={
        "ElasticNet": ":.3f",
        "RF": ":.3f",
        "shift": ":.3f",
        "abs_shift": ":.3f",
        "change_type": True,
    },
    color_discrete_map={
        "↑ More important in RF": "red",
        "↓ More important in EN": "blue",
        "No change": "gray",
    },
    title="Normalized Feature Importance Shift: Elastic Net vs Random Forest",
    labels={
        "ElasticNet": "Elastic Net (normalized importance)",
        "RF": "Random Forest (normalized importance)",
    },
)

# Add diagonal line y=x
fig.add_shape(
    type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", dash="dash")
)

fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")))

fig.update_layout(
    width=950,
    height=750,
    legend_title="Shift Type",
)

# Add vertical annotations (90° rotated, BELOW with margin)
for feat, row in top_shift_features.iterrows():
    fig.add_annotation(
        x=row["ElasticNet"],
        y=row["RF"],
        text=feat,
        textangle=90,
        showarrow=False,
        font=dict(size=10, color="black"),
        xanchor="center",
        yanchor="top",
        yshift=-15,
    )

# --- Paths ---
png_path = "results/feature_importance_shift_scatter.png"
html_path = "results/feature_importance_shift_scatter.html"

# Save PNG
fig.write_image(png_path, scale=2)

# Create descriptive text
description_text = f"""
<h2>What this scatter plot shows</h2>
<p>
Each point represents a single feature used in the models. The x-axis shows its <b>normalized importance in Elastic Net</b> 
(a linear model), while the y-axis shows its <b>normalized importance in Random Forest</b> 
(a non-linear model).<br><br>

- Points along the <b>diagonal dashed line</b> have similar importance in both models.<br>
- <span style="color:red;"><b>Red points</b></span> gained importance in the non-linear model (RF).<br>
- <span style="color:blue;"><b>Blue points</b></span> were more important in the linear model (EN).<br>
- Point size reflects how strongly the importance shifted between models.<br><br>

Hover over any point to see the feature name and exact values.<br>
The <b>top {TOP_HIGHLIGHT} most shifted features</b> are annotated directly on the plot.
</p>
"""

# Embed figure HTML inside a simple 2-column layout
fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

full_html = f"""
<html>
<head>
<title>Feature Importance Shift</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
}}
.container {{
    display: flex;
    flex-direction: row;
}}
.text-panel {{
    width: 30%;
    padding-right: 20px;
}}
.plot-panel {{
    width: 70%;
}}
</style>
</head>
<body>
<div class="container">
    <div class="text-panel">
        {description_text}
    </div>
    <div class="plot-panel">
        {fig_html}
    </div>
</div>
</body>
</html>
"""

# Save combined HTML
with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"Saved PNG: {png_path}")
print(f"Saved interactive HTML with description: {html_path}")

# Show interactive plot in current session
fig.show()
