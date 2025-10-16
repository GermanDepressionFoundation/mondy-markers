import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

DATA_DIR = "results_with_nightfeatures_perfeaturescaler_timeaware_test"


def extract_pid(path):
    base = os.path.basename(path)
    m = re.match(r"(.+?)_(EN|RF)_perm_group_summary\.csv$", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def load_model_files(model_tag, data_dir=DATA_DIR):
    pattern = os.path.join(data_dir, f"*_{model_tag}_perm_group_summary.csv")
    files = sorted(glob.glob(pattern))
    records = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue

        # Flexible column handling
        feature_col = (
            "group"
            if "group" in df.columns
            else ("feature" if "feature" in df.columns else None)
        )
        rel_mae_col = None
        for cand in [
            "rel_delta_maE",
            "rel_delta_MAE",
            "rel_delta_mae",
            "rel_delta_Mae",
        ]:
            if cand in df.columns:
                rel_mae_col = cand
                break

        if feature_col is None or rel_mae_col is None:
            print(
                f"Skipping {fp}: required columns not found. Found columns: {list(df.columns)}"
            )
            continue

        pid = extract_pid(fp)
        tmp = df[[feature_col, rel_mae_col]].copy()
        tmp.columns = ["feature", "rel_delta_mae"]
        tmp["pid"] = pid
        records.append(tmp)

    if not records:
        return pd.DataFrame(columns=["feature", "rel_delta_mae", "pid"])
    return pd.concat(records, ignore_index=True)


# ---------- NEW: compute a shared feature order across EN & RF ----------
def compute_shared_feature_order(df_en, df_rf):
    """
    Build a single y-axis (feature) order used by both EN and RF.
    Strategy: for each feature, take the median rel_delta_mae across participants
    within each model; then take the mean of the available medians across models;
    sort descending.
    """

    def feature_medians(df):
        if df.empty:
            return pd.Series(dtype=float)
        pivot = df.pivot_table(
            index="feature", values="rel_delta_mae", aggfunc="median"
        )
        return pivot["rel_delta_mae"]

    m_en = feature_medians(df_en)
    m_rf = feature_medians(df_rf)

    # union of features
    all_features = pd.Index(sorted(set(m_en.index).union(set(m_rf.index))))
    # align medians, then average (ignore NaNs)
    merged = pd.concat([m_en.reindex(all_features), m_rf.reindex(all_features)], axis=1)
    merged.columns = ["EN_med", "RF_med"]
    combined_score = merged.mean(axis=1, skipna=True)

    # If everything is NaN (edge case), just return alphabetical
    if combined_score.isna().all():
        return list(all_features)

    # Sort by combined score (desc), then alphabetically to stabilize ties
    order = combined_score.sort_values(ascending=False).index.tolist()
    return order


# ---------- UPDATED: accept shared order in plotting ----------
def make_heatmap(df_tidy, model_tag, outpath_png, feature_order=None):
    if df_tidy.empty:
        print(f"No data for model {model_tag}; skipping figure.")
        return

    pivot = df_tidy.pivot_table(
        index="feature", columns="pid", values="rel_delta_mae", aggfunc="median"
    )

    # enforce shared order (add missing rows with NaN so the y-axes align)
    if feature_order is not None:
        missing = [f for f in feature_order if f not in pivot.index]
        if missing:
            pivot = pd.concat([pivot, pd.DataFrame(index=missing)], axis=0)
        pivot = pivot.loc[feature_order]
    else:
        # fallback: model-specific order
        feature_order_local = (
            pivot.median(axis=1).sort_values(ascending=False).index.tolist()
        )
        pivot = pivot.loc[feature_order_local]

    fig, ax = plt.subplots(
        figsize=(max(6, pivot.shape[1] * 0.5), max(4, pivot.shape[0] * 0.35))
    )
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    # if pivot is all-NaN, guard limits
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    max_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="PuOr_r",  # colorblind-safe diverging colormap
        vmin=-max_abs,  # symmetric color limits
        vmax=max_abs,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative feature importance")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Participant (pid)")
    ax.set_ylabel("Feature (group)")
    ax.set_title(f"{model_tag}: Relative feature importance across participants")

    if pivot.size <= 400:
        import pandas as pd

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Binning logic (unchanged) ----
def bin_equal_width(values, n_bins=4):
    max_abs = np.nanmax(np.abs(values)) if np.size(values) else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        edges = np.array([0, 1, 2, 3, 4], dtype=float)
    else:
        edges = np.linspace(0, max_abs, n_bins + 1)
    labels = ["No influence", "Low", "Moderate", "High"]
    return edges, labels


def plot_categorical_grid(df_tidy, model_tag, out_png, feature_order=None):
    if df_tidy.empty:
        print(f"No data for {model_tag}; skipping.")
        return

    mags = np.abs(df_tidy["rel_delta_mae"].values.astype(float))
    edges, labels = bin_equal_width(mags, n_bins=4)

    cats = np.digitize(np.abs(df_tidy["rel_delta_mae"].values), edges, right=False) - 1
    cats = np.clip(cats, 0, 3)
    df_tidy = df_tidy.copy()
    df_tidy["category_idx"] = cats
    df_tidy["category_label"] = [labels[i] for i in cats]

    pivot = df_tidy.pivot_table(
        index="feature", columns="pid", values="category_idx", aggfunc="median"
    )

    # enforce shared order (add missing rows to align)
    if feature_order is not None:
        missing = [f for f in feature_order if f not in pivot.index]
        if missing:
            pivot = pd.concat([pivot, pd.DataFrame(index=missing)], axis=0)
        pivot = pivot.loc[feature_order]
    else:
        order = pivot.median(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[order]

    # Use a gradual color map (colorblind-safe sequential)
    cmap = cm.get_cmap("PuBuGn", 4)  # 4 evenly spaced colors
    norm = Normalize(vmin=0, vmax=3)

    # Draw
    fig, ax = plt.subplots(
        figsize=(max(6, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.4))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

    # Axes
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_tag} — Relative Feature Importance (binned)")

    colors = [cmap(i / 3) for i in range(4)]  # evenly spaced along the colormap
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=labels[i]) for i in range(4)
    ]
    ax.legend(
        handles=legend_handles,
        title="Importance",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# Optional: a colorblind-friendly palette (Okabe–Ito, repeated if needed)
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]


def stacked_positive_relmae(
    df_tidy,
    model_tag,
    out_png,
    feature_order=None,  # pass shared order to align EN/RF
    top_k=None,  # e.g., 12 features to keep plot readable (by total positive contribution)
    highlight_top_n=3,  # visually emphasize top-N features overall
    label_totals=True,  # print total per bar on top
    sort_participants_by_total=True,  # sort participants by total height
):
    """
    Stacked bar plot per participant with only positive relative MAE contributions.
    - Aggregates median rel_delta_mae per (feature, pid)
    - Clips negatives to zero
    - No normalization (keeps absolute comparability)
    - Labels total per participant
    - Highlights globally top-N features with hatch + edgecolor
    """
    if df_tidy.empty:
        print(f"No data for {model_tag}; skipping stacked bar.")
        return

    # 1) Aggregate to median per (feature, pid), clip to positive
    agg = (
        df_tidy.groupby(["feature", "pid"], as_index=False)["rel_delta_mae"]
        .median()
        .rename(columns={"rel_delta_mae": "rel_pos"})
    )
    agg["rel_pos"] = agg["rel_pos"].clip(lower=0.0)

    # 2) Pivot to features x participants (sum in case multiple rows remain)
    pivot = agg.pivot_table(
        index="feature", columns="pid", values="rel_pos", aggfunc="sum"
    ).fillna(0.0)

    # 3) Optional: reduce to top-K features (by total positive contribution)
    if top_k is not None and top_k < pivot.shape[0]:
        totals_feat = pivot.sum(axis=1)
        keep = totals_feat.sort_values(ascending=False).head(top_k).index
        pivot = pivot.loc[keep]

    # 4) Enforce feature order if provided; else order by total contribution
    if feature_order is not None:
        available = [f for f in feature_order if f in pivot.index]
        pivot = pivot.reindex(available)
    else:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    # 5) Optionally sort participants by total bar height (descending)
    totals_pid = pivot.sum(axis=0)
    if sort_participants_by_total:
        pivot = pivot.loc[:, totals_pid.sort_values(ascending=False).index]
        totals_pid = pivot.sum(axis=0)

    # 6) Determine globally top-N features to highlight
    global_feat_totals = pivot.sum(axis=1).sort_values(ascending=False)
    highlighted = set(global_feat_totals.head(max(0, int(highlight_top_n))).index)

    # 7) Plot
    pids = pivot.columns.tolist()
    features = pivot.index.tolist()
    x = np.arange(len(pids))

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(pids)), max(5, 0.35 * len(features)))
    )

    bottoms = np.zeros(len(pids))
    colors = (OKABE_ITO * ((len(features) // len(OKABE_ITO)) + 1))[: len(features)]

    bars_by_feat = {}
    for i, feat in enumerate(features):
        heights = pivot.loc[feat].values
        is_highlight = feat in highlighted
        bars = ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=feat,
            color=colors[i],
            edgecolor="black" if is_highlight else None,
            linewidth=1.2 if is_highlight else 0.0,
            hatch="///" if is_highlight else None,
        )
        bars_by_feat[feat] = bars
        bottoms += heights

    # 8) Label totals on top (if any non-zero)
    if label_totals:
        for xi, total in zip(x, totals_pid.values):
            if total > 0:
                ax.text(xi, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    # Axes and labels
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=45, ha="right")
    ax.set_xlabel("Participant (pid)")
    ax.set_ylabel("Relative feature importance (sum of positives)")
    ax.set_title(
        f"{model_tag}: Stacked positive relative feature importance per participant"
    )

    # Legend outside for readability
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------- MAIN: build shared order, then plot both models ----------------
outputs = []

df_en = load_model_files("EN")
df_rf = load_model_files("RF")

# Save combined data
for model, df in [("EN", df_en), ("RF", df_rf)]:
    combined_path = os.path.join(DATA_DIR, f"{model}_rel_delta_mae_combined.csv")
    df.to_csv(combined_path, index=False)
    outputs.append(combined_path)

# Compute one shared y-axis order across EN and RF
shared_order = compute_shared_feature_order(df_en, df_rf)

# Plot continuous heatmaps with the SAME y order
for model, df in [("EN", df_en), ("RF", df_rf)]:
    fig_path = os.path.join(DATA_DIR, f"{model}_rel_delta_mae_heatmap.png")
    make_heatmap(df, model, fig_path, feature_order=shared_order)
    outputs.append(fig_path)

# Plot categorical (binned) heatmaps with the SAME y order
for model, df in [("EN", df_en), ("RF", df_rf)]:
    out = os.path.join(DATA_DIR, f"{model}_rel_delta_mae_categorical_heatmap.png")
    plot_categorical_grid(df, model, out, feature_order=shared_order)
    outputs.append(out)

for model, df in [("EN", df_en), ("RF", df_rf)]:
    out = os.path.join(DATA_DIR, f"{model}_rel_delta_mae_stacked_barplot.png")
    stacked_positive_relmae(
        df_tidy=df,
        model_tag=model,
        out_png=out,
        feature_order=shared_order,
        top_k=12,
        highlight_top_n=3,  # hatch the 3 most important features globally
        label_totals=True,
        sort_participants_by_total=False,
    )
    outputs.append(out)

print(outputs)
