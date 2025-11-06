import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

DATA_DIR = "results_dummycomparison_timeaware_5050_split_v10"

# ===== Consistent styles for stacked bars (up to 15 groups) =====
COLORS_15 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]

HATCHES_15 = [
    "/",
    "\\",
    "|",
    "-",
    "+",
    "x",
    "o",
    "O",
    ".",
    "*",
    "///",
    "\\\\\\",
    "||",
    "--",
    "xx",
]


def build_style_map(groups, preferred_order=None):
    """
    Create a stable (order, color_map, hatch_map) for the given group names.
    - preferred_order: list to pin a canonical order; others follow alphabetically.
    """
    groups = list(dict.fromkeys(groups))  # keep unique, stable
    if preferred_order:
        base = [g for g in preferred_order if g in groups]
        tail = sorted([g for g in groups if g not in base])
        order = base + tail
    else:
        order = sorted(groups)

    color_map = {g: COLORS_15[i % len(COLORS_15)] for i, g in enumerate(order)}
    hatch_map = {g: HATCHES_15[i % len(HATCHES_15)] for i, g in enumerate(order)}
    return order, color_map, hatch_map


def extract_pid(path):
    """
    Extract pid from either:
      <pid>_(EN|RF)_perm_group_summary.csv
      <pid>_(EN|RF)_perm_group_folds.csv
    Fallback: filename stem.
    """
    base = os.path.basename(path)
    m = re.match(r"(.+?)_(EN|RF)_perm_group_(summary|folds)\.csv$", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def std_to_width(std_vals, s_lo, s_hi, w_min=0.35, w_max=0.90):
    """
    Map std to width: higher std -> narrower segment.
    std_vals: array-like of shape (n_folds,)
    """
    s = (std_vals - s_lo) / (s_hi - s_lo)
    s = np.clip(s, 0.0, 1.0)
    # invert so higher std -> smaller width
    return w_min + s * (w_max - w_min)


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
        rel_mae_col = "rel_delta_MAE"
        delta_mae_std_col = "delta_MAE_std"

        pid = extract_pid(fp)
        tmp = df[[feature_col, rel_mae_col, delta_mae_std_col]].copy()
        tmp.columns = ["feature", "rel_delta_mae", "delta_mae_std"]
        tmp["pid"] = pid
        records.append(tmp)

    if not records:
        return pd.DataFrame(
            columns=["feature", "rel_delta_mae", "delta_mae_std", "pid"]
        )
    return pd.concat(records, ignore_index=True)


def _coerce_fold_num(s):
    """Try to coerce fold labels to numeric; return NaN if not possible."""
    return pd.to_numeric(s, errors="coerce")


def load_model_files_from_last_fold(model_tag, data_dir=DATA_DIR):
    """
    Load all <pid>_{model_tag}_perm_group_folds.csv and return a tidy DF:
        columns = ["feature", "rel_delta_mae", "pid"]
    using ONLY the *last* fold per participant (max numeric fold if available).
    """
    pattern = os.path.join(data_dir, f"*_{model_tag}_perm_group_folds.csv")
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
        fold_col = "fold" if "fold" in df.columns else None
        rel_mae_col = None
        for cand in [
            "rel_delta_MAE",
            "rel_delta_maE",
            "rel_delta_mae",
            "rel_delta_Mae",
        ]:
            if cand in df.columns:
                rel_mae_col = cand
                break

        if feature_col is None or rel_mae_col is None or fold_col is None:
            print(
                f"Skipping {fp}: required columns not found. Columns: {list(df.columns)}"
            )
            continue

        # Determine last fold
        df["_fold_num"] = _coerce_fold_num(df[fold_col])
        if df["_fold_num"].notna().any():
            last_fold_val = df.loc[df["_fold_num"].idxmax(), fold_col]
        else:
            # Fall back to lexicographic last if non-numeric
            last_fold_val = df[fold_col].astype(str).sort_values().iloc[-1]

        df_last = df[df[fold_col] == last_fold_val].copy()

        pid = extract_pid(fp)
        tmp = df_last[[feature_col, rel_mae_col]].copy()
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
    """
    Categorical (binned) heatmap across participants using ONLY non-negative
    relative ΔMAE values (negative values are set to 0).
    Bins: No influence / Low / Moderate / High (equidistant on [0, max]).
    """
    if df_tidy.empty:
        print(f"No data for {model_tag}; skipping.")
        return

    tidy = df_tidy.copy()
    # 1) Keep only positive contributions (set negatives to 0)
    tidy["rel_delta_mae"] = tidy["rel_delta_mae"].clip(lower=0.0)

    # 2) Build equidistant bins from non-negative values
    mags = tidy["rel_delta_mae"].values.astype(float)
    edges, labels = bin_equal_width(mags, n_bins=4)

    # Edge case: if everything is zero/NaN, force a simple 0..1 binning so we still render
    if (np.nanmax(mags) if mags.size else 0.0) == 0.0:
        edges = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)

    # 3) Discretize (0..3)
    cats = np.digitize(tidy["rel_delta_mae"].values, edges, right=False) - 1
    cats = np.clip(cats, 0, 3)
    tidy["category_idx"] = cats
    tidy["category_label"] = [labels[i] for i in cats]

    # 4) Pivot to (feature x pid) with median category index
    pivot = tidy.pivot_table(
        index="feature", columns="pid", values="category_idx", aggfunc="median"
    )

    # 5) Enforce shared feature order if provided
    if feature_order is not None:
        missing = [f for f in feature_order if f not in pivot.index]
        if missing:
            pivot = pd.concat([pivot, pd.DataFrame(index=missing)], axis=0)
        pivot = pivot.loc[feature_order]
    else:
        order = pivot.median(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[order]

    # 6) Gradual, colorblind-safe sequential colormap (4 discrete steps)
    cmap = cm.get_cmap("PuBuGn", 4)
    norm = Normalize(vmin=0, vmax=3)

    # 7) Plot
    fig, ax = plt.subplots(
        figsize=(max(6, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.4))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_tag} — Relative Feature Importance (binned, ≥0 only)")

    # Legend (plain language)
    colors = [cmap(i / 3) for i in range(4)]
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=labels[i]) for i in range(4)
    ]
    ax.legend(
        handles=legend_handles,
        title="Importance (≥0)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def stacked_positive_relmae(
    df_tidy,
    model_tag,
    out_png,
    feature_order=None,  # pass shared order to align EN/RF
    top_k=None,  # e.g., 12 features to keep plot readable (by total positive contribution)
    label_totals=True,  # print total per bar on top
    sort_participants_by_total=True,  # sort participants by total height
    style_order=None,
    color_map=None,
    hatch_map=None,
):
    """
    Stacked bar plot per participant with only positive relative MAE contributions.
    - Aggregates median rel_delta_mae per (feature, pid)
    - Clips negatives to zero
    - No normalization (keeps absolute comparability)
    - Labels total per participant
    """
    if df_tidy.empty:
        print(f"No data for {model_tag}; skipping stacked bar.")
        return

    # 1) Aggregate to median per (feature, pid), clip to positive
    agg = (
        df_tidy.groupby(["feature", "pid"], as_index=False)["rel_delta_mae"]
        .mean()
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

    # 7) Plot
    pids = pivot.columns.tolist()
    features = pivot.index.tolist()
    x = np.arange(len(pids))

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(pids)), max(5, 0.35 * len(features)))
    )

    # --- Use shared styles ---
    style_order = style_order or STYLE_ORDER
    color_map = color_map or COLOR_MAP
    hatch_map = hatch_map or HATCH_MAP

    # Plot features in the canonical style order, filtered to what's present
    plot_features = [f for f in style_order if f in features]

    bottoms = np.zeros(len(pids))
    bars_by_feat = {}
    for feat in plot_features:
        heights = pivot.loc[feat].values
        bars = ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=feat,
            color=color_map.get(feat, "#cccccc"),
            edgecolor="black",
            hatch=hatch_map.get(feat, ""),
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


def stacked_positive_relmae_with_std(
    df_tidy,
    model_tag,
    out_png,
    feature_order=None,  # pass shared order to align EN/RF
    top_k=None,  # e.g., 12 features to keep plot readable (by total positive contribution)
    label_totals=True,  # print total per bar on top
    sort_participants_by_total=True,  # sort participants by total height
    style_order=None,
    color_map=None,
    hatch_map=None,
):
    """
    Stacked bar plot per participant with only positive relative MAE contributions.
    - Aggregates median rel_delta_mae per (feature, pid)
    - Clips negatives to zero
    - No normalization (keeps absolute comparability)
    - Labels total per participant
    """
    if df_tidy.empty:
        print(f"No data for {model_tag}; skipping stacked bar.")
        return

    # 1) Aggregate to median per (feature, pid), clip to positive
    agg = (
        df_tidy.groupby(["feature", "pid"], as_index=False)["rel_delta_mae"]
        .mean()
        .rename(columns={"rel_delta_mae": "rel_pos"})
    )
    agg["rel_pos"] = agg["rel_pos"].clip(lower=0.0)

    # 2) Pivot to features x participants (sum in case multiple rows remain)
    pivot = agg.pivot_table(
        index="feature", columns="pid", values="rel_pos", aggfunc="sum"
    ).fillna(0.0)

    # --- NEW: std per (feature, fold) and pivot to align with `piv`
    std_agg = (
        df_tidy.groupby(["feature", "pid"], as_index=False)["delta_mae_std"]
        .median()  # or .mean(), whichever matches your intent
        .rename(columns={"delta_mae_std": "std_val"})
    )

    std_piv = std_agg.pivot_table(
        index="feature", columns="pid", values="std_val", aggfunc="mean"
    ).fillna(0.0)

    # Robust global normalization for std -> width mapping
    std_all = std_piv.values.ravel()
    std_all = std_all[np.isfinite(std_all)]
    if len(std_all) == 0:
        s_lo, s_hi = 0.0, 1.0
    else:
        # 5–95% robust range to avoid outliers dominating
        s_lo, s_hi = np.percentile(std_all, [5, 95])
        if s_hi <= s_lo:
            s_hi = s_lo + 1e-12  # avoid divide by zero

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

    # 7) Plot
    pids = pivot.columns.tolist()
    features = pivot.index.tolist()
    x = np.arange(len(pids))

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(pids)), max(5, 0.35 * len(features)))
    )

    # --- Use shared styles ---
    style_order = style_order or STYLE_ORDER
    color_map = color_map or COLOR_MAP
    hatch_map = hatch_map or HATCH_MAP

    plot_features = [f for f in style_order if f in features]

    bottoms = np.zeros(len(pids))
    bars_by_feat = {}
    for feat in plot_features:
        heights = pivot.loc[feat].values
        std_vals = (
            std_piv.loc[feat].values
            if feat in std_piv.index
            else np.zeros_like(heights)
        )
        std_vals = np.where(heights > 0, std_vals, 0.0)
        widths = std_to_width(std_vals, s_lo, s_hi)
        bars = ax.bar(
            x,
            heights,
            width=widths,
            bottom=bottoms,
            label=feat,
            color=color_map.get(feat, "#cccccc"),
            edgecolor="black",
            hatch=hatch_map.get(feat, ""),
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

# Build a single style map from the union of all features across EN & RF
all_features_for_style = sorted(
    set(df_en["feature"].dropna().unique()).union(
        set(df_rf["feature"].dropna().unique())
    )
)
STYLE_ORDER, COLOR_MAP, HATCH_MAP = build_style_map(
    all_features_for_style,
    preferred_order=shared_order,  # keeps style order aligned to your shared y-order
)

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
        label_totals=True,
        sort_participants_by_total=False,
        style_order=STYLE_ORDER,
        color_map=COLOR_MAP,
        hatch_map=HATCH_MAP,
    )
    outputs.append(out)

    out = os.path.join(DATA_DIR, f"{model}_rel_delta_mae_stacked_barplot_std.png")
    stacked_positive_relmae_with_std(
        df_tidy=df,
        model_tag=model,
        out_png=out,
        feature_order=shared_order,
        top_k=12,
        label_totals=True,
        sort_participants_by_total=False,
        style_order=STYLE_ORDER,
        color_map=COLOR_MAP,
        hatch_map=HATCH_MAP,
    )
    outputs.append(out)

print(outputs)

# ====================== PER-PARTICIPANT, PER-FOLD PLOTS ======================


def load_fold_metrics(pid, model_tag, data_dir=DATA_DIR):
    """
    Load <pid>_<model>_fold_metrics.csv and return:
      { str(fold): {"r2": float|None, "n_test_samples": int|None} }
    """
    path = os.path.join(data_dir, f"{pid}_{model_tag}_fold_metrics.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping fold metrics for {pid} {model_tag}: {e}")
        return {}

    if "fold" not in df.columns or "r2" not in df.columns:
        print(f"Fold metrics missing required columns in {path}: {list(df.columns)}")
        return {}

    df["fold"] = df["fold"].astype(str)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")

    # robust detection of n_test_samples
    n_col = None
    for cand in ["n_test_samples", "n_test", "n_test_days"]:
        if cand in df.columns:
            n_col = cand
            df[n_col] = pd.to_numeric(df[n_col], errors="coerce")
            break

    out = {}
    for _, row in df.iterrows():
        out[row["fold"]] = {
            "r2": float(row["r2"]) if pd.notna(row["r2"]) else None,
            "n_test_samples": (
                int(row[n_col]) if n_col and pd.notna(row[n_col]) else None
            ),
        }
    return out


def load_per_fold_file(model_tag, data_dir=DATA_DIR):
    """
    Load all <pid>_{model_tag}_perm_group_folds.csv files and return a dict:
        { pid: tidy_df }
    with columns: feature, fold, rel_delta_mae
    """
    pattern = os.path.join(data_dir, f"*_{model_tag}_perm_group_folds.csv")
    files = sorted(glob.glob(pattern))
    pid_to_df = {}
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue

        # flexible columns
        feature_col = (
            "group"
            if "group" in df.columns
            else ("feature" if "feature" in df.columns else None)
        )
        fold_col = "fold" if "fold" in df.columns else None
        rel_delta_mae_col = "rel_delta_MAE"
        delta_mae_std_col = "delta_MAE_std"

        pid = extract_pid(fp)
        tmp = df[[feature_col, fold_col, rel_delta_mae_col, delta_mae_std_col]].copy()
        tmp.columns = ["feature", "fold", "rel_delta_mae", "delta_mae_std"]
        # Ensure fold is sortable: try numeric, else leave as string
        try:
            tmp["fold_num"] = pd.to_numeric(tmp["fold"], errors="coerce")
        except Exception:
            tmp["fold_num"] = np.nan
        pid_to_df[pid] = tmp
    return pid_to_df


def _order_folds(df):
    """
    Decide a sensible fold order:
    - If any numeric values parse, sort by numeric (NaNs last, then by string).
    - Otherwise, sort by string representation.
    """
    has_num = df["fold_num"].notna().any()
    if has_num:
        # For rows where fold_num is NaN, keep their original label order after numeric ones
        folds_num = df.drop_duplicates("fold")[["fold", "fold_num"]].sort_values(
            ["fold_num", "fold"], na_position="last"
        )
        ordered = folds_num["fold"].tolist()
    else:
        ordered = sorted(df["fold"].astype(str).unique(), key=lambda x: (len(x), x))
    return ordered


def per_participant_heatmap_continuous(df_pid, pid, model_tag, out_png):
    """
    Groups x Folds heatmap using rel_delta_mae (median per (feature, fold)).
    Diverging colormap centered at 0.
    """
    if df_pid.empty:
        return
    # aggregate
    piv = (
        df_pid.groupby(["feature", "fold"], as_index=False)["rel_delta_mae"]
        .median()
        .pivot(index="feature", columns="fold", values="rel_delta_mae")
    )
    # order folds
    fold_order = _order_folds(df_pid)
    for f in fold_order:
        if f not in piv.columns:
            piv[f] = np.nan
    piv = piv[fold_order]
    # order features by median across folds
    feat_order = piv.median(axis=1).sort_values(ascending=False).index.tolist()
    piv = piv.loc[feat_order]

    fig, ax = plt.subplots(
        figsize=(max(6, 0.4 * len(fold_order)), max(4, 0.35 * len(feat_order)))
    )
    vmin, vmax = np.nanmin(piv.values), np.nanmax(piv.values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    max_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(
        piv.values, aspect="auto", cmap="PuOr_r", vmin=-max_abs, vmax=max_abs
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative feature importance")

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Feature (group)")
    ax.set_title(f"{model_tag} — {pid}: Relative feature importance by fold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def per_participant_heatmap_categorical(df_pid, pid, model_tag, out_png):
    """
    Groups x Folds categorical/binned heatmap (No/Low/Moderate/High),
    using only non-negative relative ΔMAE values (negative values are set to 0).
    Gradual color map shows increasing positive importance.
    """
    if df_pid.empty:
        return

    # Copy and clip negative values to 0
    tidy = df_pid.copy()
    tidy["rel_delta_mae"] = tidy["rel_delta_mae"].clip(lower=0.0)

    # Bin edges from non-negative magnitudes (equidistant)
    mags = tidy["rel_delta_mae"].values.astype(float)
    edges, labels = bin_equal_width(mags, n_bins=4)

    cats = np.digitize(tidy["rel_delta_mae"].values, edges, right=False) - 1
    cats = np.clip(cats, 0, 3)
    tidy["category_idx"] = cats
    tidy["category_label"] = [labels[i] for i in cats]

    # Pivot: features x folds (median category index)
    piv = (
        tidy.groupby(["feature", "fold"], as_index=False)["category_idx"]
        .median()
        .pivot(index="feature", columns="fold", values="category_idx")
    )

    # Ensure all folds exist and order them
    fold_order = _order_folds(tidy)
    for f in fold_order:
        if f not in piv.columns:
            piv[f] = np.nan
    piv = piv[fold_order]

    # Order features by median category (most important at top)
    feat_order = piv.median(axis=1).sort_values(ascending=False).index.tolist()
    piv = piv.loc[feat_order]

    # Gradual color map (colorblind-safe sequential)
    cmap = cm.get_cmap("PuBuGn", 4)
    norm = Normalize(vmin=0, vmax=3)

    # --- Plot ---
    fig, ax = plt.subplots(
        figsize=(max(6, 0.4 * len(fold_order)), max(4, 0.35 * len(feat_order)))
    )
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Feature (group)")
    ax.set_title(f"{model_tag} — {pid}: Relative Feature Importance (binned, ≥0 only)")

    # Legend with plain-language categories
    colors = [cmap(i / 3) for i in range(4)]
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=labels[i]) for i in range(4)
    ]
    ax.legend(
        handles=legend_handles,
        title="Importance (≥0)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def per_participant_stacked_positive(
    df_pid,
    pid,
    model_tag,
    out_png,
    top_k=None,
    label_totals=True,
    style_order=None,
    color_map=None,
    hatch_map=None,
):
    """
    Stacked bars per fold (x-axis), stacking only positive rel_delta_mae per feature.
    Absolute scale (no normalization). Totals labeled.
    Now also annotates each bar with the fold R² from <pid>_<model>_fold_metrics.csv.
    """
    if df_pid.empty:
        return

    # median per (feature, fold), clip negatives
    agg = (
        df_pid.groupby(["feature", "fold"], as_index=False)["rel_delta_mae"]
        .median()
        .rename(columns={"rel_delta_mae": "rel_pos"})
    )
    agg["rel_pos"] = agg["rel_pos"].clip(lower=0.0)

    # pivot: features x folds
    piv = agg.pivot_table(
        index="feature", columns="fold", values="rel_pos", aggfunc="sum"
    ).fillna(0.0)

    # order folds (use your helper) and ensure all exist
    fold_order = _order_folds(df_pid)
    for f in fold_order:
        if f not in piv.columns:
            piv[f] = 0.0
    piv = piv[fold_order]

    # top-k features (by total positive)
    if top_k is not None and top_k < piv.shape[0]:
        totals_feat = piv.sum(axis=1)
        keep = totals_feat.sort_values(ascending=False).head(top_k).index
        piv = piv.loc[keep]

    # feature order by total positive
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]

    # totals per fold
    totals_fold = piv.sum(axis=0)

    # --- NEW: load fold-level R² and align to fold_order
    metrics_map = load_fold_metrics(pid, model_tag, DATA_DIR)
    # keys as strings for robust matching
    r2_per_fold = [metrics_map.get(str(f), None) for f in fold_order]

    x = np.arange(len(fold_order))
    features = piv.index.tolist()
    fig, ax = plt.subplots(
        figsize=(max(7, 0.7 * len(fold_order)), max(5, 0.35 * len(features)))
    )

    style_order = style_order or STYLE_ORDER
    color_map = color_map or COLOR_MAP
    hatch_map = hatch_map or HATCH_MAP

    plot_features = [f for f in style_order if f in features]

    bottoms = np.zeros(len(fold_order))
    for feat in plot_features:
        heights = piv.loc[feat].values
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=feat,
            color=color_map.get(feat, "#cccccc"),
            edgecolor="black",
            hatch=hatch_map.get(feat, ""),
        )
        bottoms += heights

    if label_totals:
        ymax = totals_fold.max() if len(totals_fold) else 0.0
        offset = 0.02 * ymax if ymax > 0 else 0.02
        for xi, f in enumerate(fold_order):
            total = totals_fold.loc[f]
            m = metrics_map.get(str(f), {})
            r2_val = m.get("r2")
            n_test = m.get("n_test_samples")

            # Build multi-line label
            lines = []
            if r2_val is not None:
                lines.append(f"R²={r2_val:.2f}")
            if n_test is not None:
                lines.append(f"n={int(n_test)}")

            label = "\n".join(lines)
            if label:
                ax.text(
                    xi,
                    total + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    linespacing=1.1,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(fold_order, rotation=45, ha="right")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Relative feature importance (sum of positives)")
    ax.set_title(
        f"{model_tag} — {pid}: Stacked positive relative feature importance by fold"
    )
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    # Add top margin for annotations (extra headroom)
    ymax = totals_fold.max() if len(totals_fold) else 0.0
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def per_participant_stacked_positive_with_std(
    df_pid,
    pid,
    model_tag,
    out_png,
    top_k=None,
    label_totals=True,
    style_order=None,
    color_map=None,
    hatch_map=None,
):
    """
    Stacked bars per fold (x-axis), stacking only positive rel_delta_mae per feature.
    Absolute scale (no normalization). Totals labeled.
    Now also annotates each bar with the fold R² from <pid>_<model>_fold_metrics.csv.
    """
    if df_pid.empty:
        return

    # median per (feature, fold), clip negatives
    agg = (
        df_pid.groupby(["feature", "fold"], as_index=False)["rel_delta_mae"]
        .median()
        .rename(columns={"rel_delta_mae": "rel_pos"})
    )
    agg["rel_pos"] = agg["rel_pos"].clip(lower=0.0)

    # pivot: features x folds
    piv = agg.pivot_table(
        index="feature", columns="fold", values="rel_pos", aggfunc="sum"
    ).fillna(0.0)

    # --- NEW: std per (feature, fold) and pivot to align with `piv`
    std_agg = (
        df_pid.groupby(["feature", "fold"], as_index=False)["delta_mae_std"]
        .median()  # or .mean(), whichever matches your intent
        .rename(columns={"delta_mae_std": "std_val"})
    )

    std_piv = std_agg.pivot_table(
        index="feature", columns="fold", values="std_val", aggfunc="mean"
    ).fillna(0.0)

    # Robust global normalization for std -> width mapping
    std_all = std_piv.values.ravel()
    std_all = std_all[np.isfinite(std_all)]
    if len(std_all) == 0:
        s_lo, s_hi = 0.0, 1.0
    else:
        # 5–95% robust range to avoid outliers dominating
        s_lo, s_hi = np.percentile(std_all, [5, 95])
        if s_hi <= s_lo:
            s_hi = s_lo + 1e-12  # avoid divide by zero

    # top-k features (by total positive)
    if top_k is not None and top_k < piv.shape[0]:
        totals_feat = piv.sum(axis=1)
        keep = totals_feat.sort_values(ascending=False).head(top_k).index
        piv = piv.loc[keep]

    # feature order by total positive
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]

    # totals per fold
    totals_fold = piv.sum(axis=0)

    # order folds (use your helper) and ensure all exist
    fold_order = _order_folds(df_pid)
    for f in fold_order:
        if f not in piv.columns:
            piv[f] = 0.0
    piv = piv[fold_order]

    # --- NEW: load fold-level R² and align to fold_order
    metrics_map = load_fold_metrics(pid, model_tag, DATA_DIR)
    # keys as strings for robust matching
    r2_per_fold = [metrics_map.get(str(f), None) for f in fold_order]

    x = np.arange(len(fold_order))
    features = piv.index.tolist()
    fig, ax = plt.subplots(
        figsize=(max(7, 0.7 * len(fold_order)), max(5, 0.35 * len(features)))
    )

    style_order = style_order or STYLE_ORDER
    color_map = color_map or COLOR_MAP
    hatch_map = hatch_map or HATCH_MAP

    plot_features = [f for f in style_order if f in features]

    bottoms = np.zeros(len(fold_order))
    for feat in plot_features:
        heights = piv.loc[feat].values
        std_vals = (
            std_piv.loc[feat].values
            if feat in std_piv.index
            else np.zeros_like(heights)
        )
        std_vals = np.where(heights > 0, std_vals, 0.0)
        widths = std_to_width(std_vals, s_lo, s_hi)

        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=widths,
            label=feat,
            color=color_map.get(feat, "#cccccc"),
            edgecolor="black",
            hatch=hatch_map.get(feat, ""),
        )
        bottoms += heights

    if label_totals:
        ymax = totals_fold.max() if len(totals_fold) else 0.0
        offset = 0.02 * ymax if ymax > 0 else 0.02
        for xi, f in enumerate(fold_order):
            total = totals_fold.loc[f]
            m = metrics_map.get(str(f), {})
            r2_val = m.get("r2")
            n_test = m.get("n_test_samples")

            # Build multi-line label
            lines = []
            if r2_val is not None:
                lines.append(f"R²={r2_val:.2f}")
            if n_test is not None:
                lines.append(f"n={int(n_test)}")

            label = "\n".join(lines)
            if label:
                ax.text(
                    xi,
                    total + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    linespacing=1.1,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(fold_order, rotation=45, ha="right")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Relative feature importance (sum of positives)")
    ax.set_title(
        f"{model_tag} — {pid}: Stacked positive relative feature importance by fold"
    )
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    # Add top margin for annotations (extra headroom)
    ymax = totals_fold.max() if len(totals_fold) else 0.0
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------- RUN PER-PARTICIPANT FOLD PLOTS FOR EN & RF ----------------
pid_to_df_en_folds = load_per_fold_file("EN", DATA_DIR)
pid_to_df_rf_folds = load_per_fold_file("RF", DATA_DIR)

# Create a subfolder for per-participant figures
per_pid_dir = os.path.join(DATA_DIR, "per_participant_folds")
os.makedirs(per_pid_dir, exist_ok=True)

for model_tag, pid_map in [("EN", pid_to_df_en_folds), ("RF", pid_to_df_rf_folds)]:
    for pid, df_pid in pid_map.items():
        # Continuous heatmap
        out1 = os.path.join(per_pid_dir, f"{pid}_{model_tag}_folds_heatmap.png")
        per_participant_heatmap_continuous(df_pid, pid, model_tag, out1)

        # Categorical (binned) heatmap
        out2 = os.path.join(per_pid_dir, f"{pid}_{model_tag}_folds_categorical.png")
        per_participant_heatmap_categorical(df_pid, pid, model_tag, out2)

        # Stacked positive bar per fold
        out3 = os.path.join(
            per_pid_dir, f"{pid}_{model_tag}_folds_stacked_positive.png"
        )
        per_participant_stacked_positive(
            df_pid,
            pid,
            model_tag,
            out3,
            top_k=12,
            label_totals=True,
            style_order=STYLE_ORDER,
            color_map=COLOR_MAP,
            hatch_map=HATCH_MAP,
        )

        # Stacked positive bar per fold with std-based widths
        out4 = os.path.join(
            per_pid_dir, f"{pid}_{model_tag}_folds_stacked_positive_std_widths.png"
        )
        per_participant_stacked_positive_with_std(
            df_pid,
            pid,
            model_tag,
            out4,
            top_k=12,
            label_totals=True,
            style_order=STYLE_ORDER,
            color_map=COLOR_MAP,
            hatch_map=HATCH_MAP,
        )
