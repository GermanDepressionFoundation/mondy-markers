import colorsys
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch

from utils import PSEUDONYM_TO_LETTER, add_logo_to_figure

DATA_DIR = "results_v11_final"

# ===== Prefix-aware, consistent styles =====
# distinct base hues (colorblind-aware-ish). these are cluster *anchors*.
BASE_HUES = [
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

# base hatch families per prefix; we increase "density" within a family
HATCH_BASES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def _hex(rgb_tuple):
    r, g, b = [max(0, min(1, c)) for c in rgb_tuple]
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _shade_variants(base_hex, k, n, s_range=(0.55, 0.85), v_range=(0.60, 0.95)):
    """
    Create k-th shade (0-index) out of n for a base color by varying saturation/value.
    Keeps hue roughly constant so items with same prefix "feel" related.
    """
    r, g, b = to_rgb(base_hex)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Evenly space within ranges; center around original s/v
    s_lo, s_hi = s_range
    v_lo, v_hi = v_range
    if n <= 1:
        s_new, v_new = s, v
    else:
        t = k / (n - 1)
        s_new = s_lo + t * (s_hi - s_lo)
        v_new = v_hi - t * (v_hi - v_lo)  # light -> dark as k increases
    rr, gg, bb = colorsys.hsv_to_rgb(h, s_new, v_new)
    return _hex((rr, gg, bb))


def _hatch_variant(base_hatch, k, n):
    """
    Pick a hatch of the same family with increasing density.
    e.g., '/', '//', '///' ... or '.', '..', '...'
    """
    if n <= 1:
        return base_hatch
    reps = 1 + min(3, int(round(1 + 2 * (k / max(1, n - 1)))))  # 1..4
    return base_hatch * reps


def build_style_map_prefix(groups, preferred_order=None):
    """
    Stable (order, color_map, hatch_map) where groups that share the first
    3 letters get related shades and hatch family.
    - preferred_order: if provided, fixes canonical legend order.
    """
    groups = list(dict.fromkeys(groups))  # unique, stable
    if preferred_order:
        head = [g for g in preferred_order if g in groups]
        tail = [g for g in groups if g not in head]
        order = head + sorted(tail)
    else:
        order = sorted(groups)

    # cluster by first 3 letters
    def pref(g):
        return (g[:3] if isinstance(g, str) and len(g) >= 3 else str(g)[:3]).lower()

    prefix_to_members = {}
    for g in order:
        prefix_to_members.setdefault(pref(g), []).append(g)

    # assign base hue + hatch family per prefix
    color_map, hatch_map = {}, {}
    prefixes = sorted(prefix_to_members.keys())
    for p_idx, p in enumerate(prefixes):
        members = prefix_to_members[p]
        base_hex = BASE_HUES[p_idx % len(BASE_HUES)]
        base_hatch = HATCH_BASES[p_idx % len(HATCH_BASES)]

        # order members stably within prefix (respect preferred_order part already in 'order')
        members = [g for g in order if g in members]

        n = len(members)
        for k, g in enumerate(members):
            color_map[g] = _shade_variants(base_hex, k, n)
            hatch_map[g] = _hatch_variant(base_hatch, k, n)

    return order, color_map, hatch_map


def sort_legend_alphanum(ax):
    """
    Sort the legend entries alphanumerically ascending by label text.
    Works for bar plots and other standard Matplotlib legend types.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    # Sort alphabetically (case-insensitive)
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0].lower())
    labels_sorted, handles_sorted = zip(*sorted_pairs)
    ax.legend(
        handles_sorted,
        labels_sorted,
        title=ax.get_legend().get_title().get_text() if ax.get_legend() else None,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncol=1,
    )


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
        rel_R2_col = "rel_delta_R2"

        pid = extract_pid(fp)
        tmp = df[[feature_col, rel_mae_col, delta_mae_std_col, rel_R2_col]].copy()
        tmp.columns = ["feature", "rel_delta_mae", "delta_mae_std", "rel_delta_R2"]
        tmp["pid"] = pid
        records.append(tmp)

    if not records:
        return pd.DataFrame(
            columns=["feature", "rel_delta_mae", "delta_mae_std", "rel_delta_R2", "pid"]
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


def _std_widths_setup(std_piv, w_min=0.35, w_max=0.90, q_lo=5, q_hi=95):
    """
    Given a pivot of std values (index x columns), compute robust
    global quantiles for mapping std -> bar width.
    Returns (s_lo, s_hi) for std_to_width.
    """
    std_all = std_piv.values.ravel()
    std_all = std_all[np.isfinite(std_all)]
    if len(std_all) == 0:
        s_lo, s_hi = 0.0, 1.0
    else:
        s_lo, s_hi = np.percentile(std_all, [q_lo, q_hi])
        if s_hi <= s_lo:
            s_hi = s_lo + 1e-12
    return s_lo, s_hi


def _mean_r2_and_nfolds(pid, model_tag, data_dir=DATA_DIR):
    """
    Compute mean R² over folds and number of folds for a participant
    using <pid>_<model_tag>_fold_metrics.csv via your existing loader.
    Returns (mean_r2_or_None, n_folds_int_or_None).
    """
    import numpy as np

    metrics = load_fold_metrics(
        pid, model_tag, data_dir
    )  # existing helper in your file
    if not metrics:
        return None, None
    r2_vals = [m.get("r2") for m in metrics.values() if m.get("r2") is not None]
    mean_r2 = float(np.mean(r2_vals)) if r2_vals else None
    n_folds = len(metrics) if metrics else None
    return mean_r2, n_folds


def across_participants_stacked_positive_relmae(
    df_tidy,
    model_tag,
    out_png,
    feature_order=None,
    top_k=None,
    label_totals=True,
    sort_participants_by_total=True,
    style_order=None,
    color_map=None,
    hatch_map=None,
    width_by_std=False,
    std_quantiles=(5, 95),
    min_rel_delta_mae=None,  # CHANGED: allow None for dynamic threshold
    rel_threshold_fraction=0.05,  # NEW: 5% of max positive value
    pseudonym_to_letter=None,  # optional mapping {pseudonym -> letter}
):
    """
    Stacked bar plot per participant (pid) with only positive relative MAE contributions.

    Features:
      - Filters out rel_delta_mae ≤ min_rel_delta_mae
      - If min_rel_delta_mae is None, a dynamic threshold is used:
        rel_threshold_fraction * max(rel_delta_mae > 0)
      - Keeps all participants (even with no visible bars)
      - Optionally uses bar width to reflect std
      - Displays mean R² and fold count (from *_fold_metrics)
      - Optionally replaces pseudonyms on x-axis via pseudonym_to_letter
    """

    if df_tidy.empty:
        print(f"No data for {model_tag}; skipping stacked bar.")
        return

    # --- Determine dynamic threshold if requested ---
    if min_rel_delta_mae is None:
        pos_vals = df_tidy.loc[df_tidy["rel_delta_mae"] > 0, "rel_delta_mae"].values
        if pos_vals.size > 0:
            max_pos = np.nanmax(pos_vals)
            min_rel_delta_mae = rel_threshold_fraction * max_pos
        else:
            min_rel_delta_mae = 0.0  # nothing positive: don't filter out anything

    # --- Remember all participants before filtering
    all_pids = df_tidy["pid"].astype(str).tolist()
    all_pids = list(dict.fromkeys(all_pids))  # unique, stable order

    # --- Apply threshold
    df_filt = df_tidy[df_tidy["rel_delta_mae"] > min_rel_delta_mae].copy()

    # --- Aggregate mean per (feature, pid)
    if not df_filt.empty:
        agg = (
            df_filt.groupby(["feature", "pid"], as_index=False)["rel_delta_mae"]
            .mean()
            .rename(columns={"rel_delta_mae": "rel_pos"})
        )
        agg["rel_pos"] = agg["rel_pos"].clip(lower=0.0)
        pivot = agg.pivot_table(
            index="feature", columns="pid", values="rel_pos", aggfunc="sum"
        )
    else:
        pivot = pd.DataFrame(
            index=pd.Index([], name="feature"), columns=pd.Index([], name="pid")
        )

    pivot = pivot.fillna(0.0)

    # --- Std pivot for widths (if enabled)
    if width_by_std:
        if not df_filt.empty and "delta_mae_std" in df_filt.columns:
            std_agg = (
                df_filt.groupby(["feature", "pid"], as_index=False)["delta_mae_std"]
                .median()
                .rename(columns={"delta_mae_std": "std_val"})
            )
            std_piv = std_agg.pivot_table(
                index="feature", columns="pid", values="std_val", aggfunc="mean"
            ).fillna(0.0)
        else:
            std_piv = pd.DataFrame(
                index=pivot.index.copy(), columns=pd.Index([], name="pid")
            ).fillna(0.0)

    # --- Ensure all participants exist as columns
    for pid in all_pids:
        if pid not in pivot.columns:
            pivot[pid] = 0.0
        if width_by_std and pid not in std_piv.columns:
            std_piv[pid] = 0.0

    pivot = pivot[all_pids]
    if width_by_std:
        std_piv = std_piv[all_pids]

    # --- Top-K features
    if top_k is not None and top_k < pivot.shape[0]:
        keep = pivot.sum(axis=1).sort_values(ascending=False).head(top_k).index
        pivot = pivot.loc[keep]
        if width_by_std:
            std_piv = std_piv.loc[keep]

    # --- Feature order
    if feature_order is not None:
        available = [f for f in feature_order if f in pivot.index]
        pivot = pivot.reindex(available)
        if width_by_std:
            std_piv = std_piv.reindex(available)
    else:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        if width_by_std:
            std_piv = std_piv.loc[pivot.index]

    # --- Sort participants by total contribution (if not all zeros)
    totals_pid = pivot.sum(axis=0)
    if sort_participants_by_total and not np.allclose(
        totals_pid.values, totals_pid.values[0]
    ):
        order = totals_pid.sort_values(ascending=False).index.tolist()
        pivot = pivot[order]
        if width_by_std:
            std_piv = std_piv[order]
        totals_pid = pivot.sum(axis=0)

    # --- Width mapping for std (if used)
    if width_by_std:
        s_lo, s_hi = _std_widths_setup(
            std_piv, q_lo=std_quantiles[0], q_hi=std_quantiles[1]
        )

    # --- Plot
    pids = pivot.columns.tolist()
    features = pivot.index.tolist()
    x = np.arange(len(pids))

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(pids)), max(5, 0.35 * max(1, len(features))))
    )

    style_order = style_order or STYLE_ORDER
    color_map = color_map or COLOR_MAP
    hatch_map = hatch_map or HATCH_MAP
    plot_features = [f for f in style_order if f in features]

    bottoms = np.zeros(len(pids))
    if plot_features:
        for feat in plot_features:
            heights = pivot.loc[feat].values
            if width_by_std:
                std_vals = std_piv.loc[feat].values
                std_vals = np.where(heights > 0, std_vals, 0.0)
                widths = std_to_width(std_vals, s_lo, s_hi)
            else:
                widths = 0.8

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
    else:
        ax.bar(x, np.zeros(len(pids)), width=0.6, color="none", edgecolor="none")

    # --- Annotations: mean R² and fold count
    if label_totals:
        r2_nf_by_pid = {
            pid: _mean_r2_and_nfolds(pid, model_tag, DATA_DIR) for pid in pids
        }

        ymax = totals_pid.max() if len(totals_pid) else 0.0
        offset = 0.03 * ymax if (np.isfinite(ymax) and ymax > 0) else 0.03
        line_gap = 0.04 * ymax if (np.isfinite(ymax) and ymax > 0) else 0.04

        for xi, pid in enumerate(pids):
            total = totals_pid.loc[pid]
            mean_r2, n_folds = r2_nf_by_pid.get(pid, (None, None))
            y_base = total if (np.isfinite(total) and total > 0) else 0.0

            # R² (red if negative)
            if mean_r2 is not None and np.isfinite(mean_r2):
                r2_color = "red" if mean_r2 < 0 else "black"
                ax.text(
                    xi,
                    y_base + offset + line_gap,
                    rf"$\overline{{R^2}}$={mean_r2:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=r2_color,
                )
            # f=<n_folds>
            if n_folds is not None:
                ax.text(
                    xi,
                    y_base + offset,
                    f"f={n_folds}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                )

    # --- X-axis labels: letters or pseudonyms
    if pseudonym_to_letter is not None:
        # Create letter labels
        x_labels = [pseudonym_to_letter.get(pid, pid) for pid in pids]

        # Sort alphanumerically by letters
        sort_order = np.argsort(x_labels)
        # Reorder everything (pivot columns, totals, and arrays)
        pivot = pivot.iloc[:, sort_order]
        if width_by_std:
            std_piv = std_piv.iloc[:, sort_order]
        pids = [pids[i] for i in sort_order]
        x_labels = [x_labels[i] for i in sort_order]
        totals_pid = totals_pid[pids]
        ax.set_xticklabels(x_labels)
    else:
        # Default labels: just use pseudonyms, no re-sorting
        x_labels = pids
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # --- Apply labels to x-axis
    ax.set_xticks(np.arange(len(pids)))

    ax.set_xlabel(
        "Participant (pid)"
        if pseudonym_to_letter is None
        else "Participant (anonymized)"
    )
    ax.set_ylabel("Relative feature importance")

    if plot_features:
        ax.legend(
            title="Feature Group", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1
        )
        sort_legend_alphanum(ax)

    # --- Headroom for labels
    ymax = totals_pid.max() if len(totals_pid) else 0.0
    if not (np.isfinite(ymax) and ymax > 0):
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.35)

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

shared_order = compute_shared_feature_order(df_en, df_rf)

# Union of features across both models for a single global mapping
all_features_for_style = sorted(
    set(df_en["feature"].dropna().unique()).union(
        set(df_rf["feature"].dropna().unique())
    )
)

STYLE_ORDER, COLOR_MAP, HATCH_MAP = build_style_map_prefix(
    all_features_for_style,
    preferred_order=shared_order,  # keeps canonical order consistent with your heatmaps
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
    out = os.path.join(
        DATA_DIR, f"{model}_pos_rel_delta_mae_greater_2p_stacked_barplot.png"
    )
    across_participants_stacked_positive_relmae(
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
        width_by_std=False,
        min_rel_delta_mae=None,
        rel_threshold_fraction=0.02,
    )
    outputs.append(out)

    out = os.path.join(
        DATA_DIR, f"{model}_pos_rel_delta_mae_greater_2p_stacked_barplot_std.png"
    )
    across_participants_stacked_positive_relmae(
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
        width_by_std=True,
        min_rel_delta_mae=None,
        rel_threshold_fraction=0.02,
    )
    outputs.append(out)

    out = os.path.join(
        DATA_DIR,
        f"{model}_pos_rel_delta_mae_greater_2p_stacked_barplot_std_with_letters.png",
    )
    across_participants_stacked_positive_relmae(
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
        width_by_std=True,
        min_rel_delta_mae=None,
        pseudonym_to_letter=PSEUDONYM_TO_LETTER,
        rel_threshold_fraction=0.02,
    )
    outputs.append(out)

print(outputs)

# ====================== PER-PARTICIPANT, PER-FOLD PLOTS ======================


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
    width_by_std=False,  # optional std-based segment widths
    std_quantiles=(5, 95),
    min_rel_delta_mae=None,  # CHANGED: None -> dynamic
    rel_threshold_fraction=0.02,  # NEW
):
    """
    Stacked bars per fold (x-axis), stacking only positive rel_delta_mae per feature.
    Enhancements:
      - Filters out rel_delta_mae ≤ min_rel_delta_mae
      - If min_rel_delta_mae is None, a dynamic threshold is used:
        rel_threshold_fraction * max(rel_delta_mae > 0) for this participant
      - Ensures all folds are on x-axis, even if nothing passes threshold
      - Optional std-based widths (delta_mae_std)
      - Annotates each bar with R² (red if negative) and n (samples)
      - No figure title; y-label shortened to 'Relative feature importance'
    """
    if df_pid.empty:
        return

    # --- Determine dynamic threshold if requested (per participant) ---
    if min_rel_delta_mae is None:
        pos_vals = df_pid.loc[df_pid["rel_delta_mae"] > 0, "rel_delta_mae"].values
        if pos_vals.size > 0:
            max_pos = np.nanmax(pos_vals)
            min_rel_delta_mae = rel_threshold_fraction * max_pos
        else:
            min_rel_delta_mae = 0.0

    # --- Keep full fold order from the raw df (before filtering)
    fold_order = _order_folds(df_pid)

    # --- Filter tiny effects
    df_filt = df_pid[df_pid["rel_delta_mae"] > min_rel_delta_mae].copy()

    # --- Aggregate positives (median across rows per feature, fold)
    if not df_filt.empty:
        agg = (
            df_filt.groupby(["feature", "fold"], as_index=False)["rel_delta_mae"]
            .median()
            .rename(columns={"rel_delta_mae": "rel_pos"})
        )
        agg["rel_pos"] = agg["rel_pos"].clip(lower=0.0)

        piv = agg.pivot_table(
            index="feature", columns="fold", values="rel_pos", aggfunc="sum"
        )
    else:
        piv = pd.DataFrame(
            index=pd.Index([], name="feature"), columns=pd.Index([], name="fold")
        )

    piv = piv.fillna(0.0)

    # --- Optional std pivot for widths
    if width_by_std:
        if not df_filt.empty and "delta_mae_std" in df_filt.columns:
            std_agg = (
                df_filt.groupby(["feature", "fold"], as_index=False)["delta_mae_std"]
                .median()
                .rename(columns={"delta_mae_std": "std_val"})
            )
            std_piv = std_agg.pivot_table(
                index="feature", columns="fold", values="std_val", aggfunc="mean"
            ).fillna(0.0)
        else:
            std_piv = pd.DataFrame(
                index=piv.index.copy(), columns=pd.Index([], name="fold")
            ).fillna(0.0)

    # --- Ensure all folds appear as columns (even if empty after filtering)
    for f in fold_order:
        if f not in piv.columns:
            piv[f] = 0.0
        if width_by_std and f not in std_piv.columns:
            std_piv[f] = 0.0

    # Reorder columns to full, canonical fold order
    piv = piv[fold_order]
    if width_by_std:
        std_piv = std_piv[fold_order]

    # --- Top-K features (by total positive across folds)
    if top_k is not None and top_k < piv.shape[0]:
        keep = piv.sum(axis=1).sort_values(ascending=False).head(top_k).index
        piv = piv.loc[keep]
        if width_by_std:
            std_piv = std_piv.loc[keep]

    # --- Feature order (descending total)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    if width_by_std:
        std_piv = std_piv.loc[piv.index]

    # --- Totals per fold
    totals_fold = piv.sum(axis=0)

    # --- Fold metrics (R², n)
    metrics_map = load_fold_metrics(
        pid, model_tag, DATA_DIR
    )  # { fold_label: {"r2":..., "n_test_samples":...} }

    # --- Plot
    x = np.arange(len(fold_order))
    features = piv.index.tolist()
    fig, ax = plt.subplots(
        figsize=(max(7, 0.7 * len(fold_order)), max(5, 0.35 * max(1, len(features))))
    )

    style_order = style_order or STYLE_ORDER
    color_map = color_map or COLOR_MAP
    hatch_map = hatch_map or HATCH_MAP
    plot_features = [f for f in style_order if f in features]

    # --- std width normalization (robust) if enabled
    if width_by_std:
        s_lo, s_hi = _std_widths_setup(
            std_piv, q_lo=std_quantiles[0], q_hi=std_quantiles[1]
        )

    bottoms = np.zeros(len(fold_order))

    if plot_features:
        for feat in plot_features:
            heights = piv.loc[feat].reindex(fold_order).values
            if width_by_std:
                std_vals = (
                    std_piv.loc[feat].reindex(fold_order).values
                    if feat in std_piv.index
                    else np.zeros_like(heights)
                )
                std_vals = np.where(heights > 0, std_vals, 0.0)
                widths = std_to_width(std_vals, s_lo, s_hi)
            else:
                widths = 0.8

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
    else:
        # nothing above threshold -> draw empty baseline bars so x-axis & annotations still show
        ax.bar(x, np.zeros(len(fold_order)), width=0.6, color="none", edgecolor="none")

    # --- Annotations: R² (red if negative) and n on top of each fold bar
    if label_totals:
        ymax = totals_fold.max() if len(totals_fold) else 0.0
        offset = 0.03 * ymax if (np.isfinite(ymax) and ymax > 0) else 0.03
        line_gap = 0.04 * ymax if (np.isfinite(ymax) and ymax > 0) else 0.04

        for xi, f in enumerate(fold_order):
            total = totals_fold.loc[f] if f in totals_fold.index else 0.0
            y_base = total if (np.isfinite(total) and total > 0) else 0.0

            m = metrics_map.get(str(f), {})
            r2_val = m.get("r2")
            n_test = m.get("n_test_samples")

            # R² line
            if r2_val is not None and np.isfinite(r2_val):
                r2_color = "red" if r2_val < 0 else "black"
                ax.text(
                    xi,
                    y_base + offset + line_gap,
                    f"R²={r2_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=r2_color,
                )
            # n line
            if n_test is not None and np.isfinite(n_test):
                ax.text(
                    xi,
                    y_base + offset,
                    f"n={int(n_test)}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                )

    # --- Axes
    ax.set_xticks(x)
    ax.set_xticklabels(fold_order, rotation=0)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Relative feature importance")  # shortened label
    # (no title)

    if plot_features:
        ax.legend(title="Feature Group", bbox_to_anchor=(1.02, 1), loc="upper left")
        sort_legend_alphanum(ax)

    # --- Headroom for labels
    ymax = totals_fold.max() if len(totals_fold) else 0.0
    if not (np.isfinite(ymax) and ymax > 0):
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.35)

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
            per_pid_dir,
            f"{pid}_{PSEUDONYM_TO_LETTER[pid]}_{model_tag}_pos_rel_delta_mae_greater_2per_stacked.png",
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
            width_by_std=False,
            min_rel_delta_mae=None,
            rel_threshold_fraction=0.02,
        )

        # Stacked positive bar per fold with std-based widths
        out4 = os.path.join(
            per_pid_dir,
            f"{pid}_{PSEUDONYM_TO_LETTER[pid]}_{model_tag}_pos_rel_delta_mae_greater_2per_stacked_std_widths.png",
        )
        per_participant_stacked_positive(
            df_pid,
            pid,
            model_tag,
            out4,
            top_k=12,
            label_totals=True,
            style_order=STYLE_ORDER,
            color_map=COLOR_MAP,
            hatch_map=HATCH_MAP,
            width_by_std=True,
            min_rel_delta_mae=None,
            rel_threshold_fraction=0.02,
        )


def stacked_mean_relmae_per_model(
    df_en_tidy,
    df_rf_tidy,
    out_png,
    style_order=None,
    color_map=None,
    hatch_map=None,
    min_rel_delta_mae=None,  # CHANGED: allow None for dynamic threshold
    rel_threshold_fraction=0.05,  # NEW: 5% of max positive value across both models
    top_k=None,  # optional: keep top-K features by total across both models
    show_legend=True,
    data_dir=DATA_DIR,  # to read *_fold_metrics.csv
    width_by_std=False,  # encode segment width from std
    std_quantiles=(5, 95),  # robust range for width mapping
    base_width=0.65,  # width when width_by_std=False
):
    """
    Two-bar stacked plot (EN, RF). Each stack level is the MEAN relative feature
    importance across participants for that model (positives only, thresholded).

    Threshold behavior:
      - If min_rel_delta_mae is not None: use it as a fixed cutoff.
      - If min_rel_delta_mae is None: compute a dynamic threshold as
        rel_threshold_fraction * max positive rel_delta_mae across both models.

    On top of each model bar:
      - R²=<mean of per-participant mean R² over folds> (red if negative)
      - p=<#participants>

    If width_by_std=True:
      - For each model & feature, compute a robust std summary: median(delta_mae_std across participants)
      - Pool both models’ feature-stds to get robust (q_lo, q_hi) via _std_widths_setup
      - Map std -> width with std_to_width
    """

    # ---- Determine dynamic threshold if requested (global across EN+RF) ----
    if min_rel_delta_mae is None:
        pos_vals = []

        for df in (df_en_tidy, df_rf_tidy):
            if df is not None and not df.empty and "rel_delta_mae" in df.columns:
                vals = df.loc[df["rel_delta_mae"] > 0, "rel_delta_mae"].values
                if vals.size > 0:
                    pos_vals.append(vals)

        if pos_vals:
            all_pos = np.concatenate(pos_vals)
            if all_pos.size > 0:
                max_pos = np.nanmax(all_pos)
                min_rel_delta_mae = rel_threshold_fraction * max_pos
            else:
                min_rel_delta_mae = 0.0
        else:
            min_rel_delta_mae = 0.0

    # ---- helpers ----
    def _per_model_feature_means_and_pids(df_tidy, thr):
        if df_tidy.empty:
            return pd.Series(dtype=float), [], pd.DataFrame()
        pids = list(dict.fromkeys(df_tidy["pid"].astype(str).tolist()))
        df = df_tidy[df_tidy["rel_delta_mae"] > thr].copy()
        if df.empty:
            return pd.Series(dtype=float), pids, pd.DataFrame()
        g = (
            df.groupby(["feature", "pid"], as_index=False)["rel_delta_mae"]
            .mean()
            .rename(columns={"rel_delta_mae": "rel_pos"})
        )
        g["rel_pos"] = g["rel_pos"].clip(lower=0.0)
        piv = g.pivot_table(
            index="feature", columns="pid", values="rel_pos", aggfunc="sum"
        ).fillna(0.0)
        f_means = piv.mean(axis=1)
        return f_means, pids, df  # return filtered df for std summaries

    def _per_model_feature_std_robust(df_filtered):
        """
        Return Series: feature -> robust std summary (median over participants).
        df_filtered must contain columns: feature, pid, delta_mae_std (after thresholding rel_delta_mae).
        """
        if df_filtered.empty or "delta_mae_std" not in df_filtered.columns:
            return pd.Series(dtype=float)
        s = (
            df_filtered.groupby(["feature", "pid"], as_index=False)["delta_mae_std"]
            .median()
            .pivot_table(
                index="feature", columns="pid", values="delta_mae_std", aggfunc="mean"
            )
            .median(axis=1)  # robust across participants
        )
        return s

    # ---- compute per-model feature means (and keep filtered df for std) ----
    en_means, en_pids, en_df_filt = _per_model_feature_means_and_pids(
        df_en_tidy, min_rel_delta_mae
    )
    rf_means, rf_pids, rf_df_filt = _per_model_feature_means_and_pids(
        df_rf_tidy, min_rel_delta_mae
    )

    # Union of features present in either model
    features_all = sorted(set(en_means.index).union(set(rf_means.index)))

    # ---- optional top-K by total across both models ----
    if top_k is not None and top_k > 0 and len(features_all) > top_k:
        totals = pd.Series(
            {f: en_means.get(f, 0.0) + rf_means.get(f, 0.0) for f in features_all}
        )
        features_keep = totals.sort_values(ascending=False).head(top_k).index.tolist()
    else:
        features_keep = features_all

    # ---- stacking order ----
    if style_order:
        plot_features = [f for f in style_order if f in features_keep]
        remaining = [f for f in features_keep if f not in plot_features]
        if remaining:
            rem_sorted = sorted(
                remaining,
                key=lambda f: en_means.get(f, 0.0) + rf_means.get(f, 0.0),
                reverse=True,
            )
            plot_features.extend(rem_sorted)
    else:
        plot_features = sorted(
            features_keep,
            key=lambda f: en_means.get(f, 0.0) + rf_means.get(f, 0.0),
            reverse=True,
        )

    # ---- heights per feature for EN/RF ----
    en_heights = np.array([float(en_means.get(f, 0.0)) for f in plot_features])
    rf_heights = np.array([float(rf_means.get(f, 0.0)) for f in plot_features])

    # ---- optional std-based widths (robust) ----
    if width_by_std:
        en_std = (
            _per_model_feature_std_robust(en_df_filt).reindex(plot_features).fillna(0.0)
        )
        rf_std = (
            _per_model_feature_std_robust(rf_df_filt).reindex(plot_features).fillna(0.0)
        )
        # Build a pseudo-pivot with two columns to leverage your existing helper
        std_piv_like = pd.DataFrame(
            {"EN": en_std.values, "RF": rf_std.values}, index=plot_features
        )
        s_lo, s_hi = _std_widths_setup(
            std_piv_like, q_lo=std_quantiles[0], q_hi=std_quantiles[1]
        )
        # Map to widths per model per feature
        en_widths = std_to_width(en_std.values, s_lo, s_hi)
        rf_widths = std_to_width(rf_std.values, s_lo, s_hi)
    else:
        en_widths = np.full(len(plot_features), base_width, dtype=float)
        rf_widths = np.full(len(plot_features), base_width, dtype=float)

    # ---- positions and figure ----
    x = np.array([0, 1])  # positions: EN, RF
    fig, ax = plt.subplots(figsize=(6.5, max(4.5, 0.35 * max(1, len(plot_features)))))

    # shared styles
    style_order_eff = style_order or plot_features
    cmap = color_map or {}
    hmap = hatch_map or {}

    bottoms_en = 0.0
    bottoms_rf = 0.0
    bars_for_legend = []

    # Plot segments in consistent feature order
    for i, feat in enumerate(style_order_eff):
        if feat not in plot_features:
            continue
        h_en = en_heights[plot_features.index(feat)]
        h_rf = rf_heights[plot_features.index(feat)]
        w_en = en_widths[plot_features.index(feat)]
        w_rf = rf_widths[plot_features.index(feat)]

        # EN segment
        be = ax.bar(
            x[0],
            h_en,
            width=w_en,
            bottom=bottoms_en,
            color=cmap.get(feat, "#cccccc"),
            edgecolor="black",
            hatch=hmap.get(feat, ""),
            label=feat,  # legend handle
        )
        # RF segment
        br = ax.bar(
            x[1],
            h_rf,
            width=w_rf,
            bottom=bottoms_rf,
            color=cmap.get(feat, "#cccccc"),
            edgecolor="black",
            hatch=hmap.get(feat, ""),
        )

        if h_en > 0 or h_rf > 0:
            bars_for_legend.append((feat, be[0]))

        bottoms_en += h_en
        bottoms_rf += h_rf

    # ---- Annotations on top: mean R² (red if negative) and p=<participants> ----
    def _mean_r2_over_participants(model_tag, pids, data_dir):
        r2_means = []
        for pid in pids:
            m = load_fold_metrics(pid, model_tag, data_dir)
            if not m:
                continue
            r2_vals = [v.get("r2") for v in m.values() if v.get("r2") is not None]
            if r2_vals:
                r2_means.append(float(np.mean(r2_vals)))
        mean_r2 = float(np.mean(r2_means)) if r2_means else None
        return mean_r2, len(pids)

    en_mean_r2, en_p = _mean_r2_over_participants("EN", en_pids, data_dir)
    rf_mean_r2, rf_p = _mean_r2_over_participants("RF", rf_pids, data_dir)

    totals = [bottoms_en, bottoms_rf]
    ymax = max(totals) if totals else 0.0
    if not (np.isfinite(ymax) and ymax > 0):
        ymax = 1.0
    offset = 0.05 * ymax
    line_gap = 0.04 * ymax

    # EN label
    y0 = totals[0]
    if en_mean_r2 is not None and np.isfinite(en_mean_r2):
        ax.text(
            x[0],
            y0 + offset + line_gap,
            rf"$\overline{{R^2}}$={en_mean_r2:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=("red" if en_mean_r2 < 0 else "black"),
        )
    ax.text(
        x[0],
        y0 + offset,
        f"p={en_p}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    # RF label
    y1 = totals[1]
    if rf_mean_r2 is not None and np.isfinite(rf_mean_r2):
        ax.text(
            x[1],
            y1 + offset + line_gap,
            rf"$\overline{{R^2}}$={rf_mean_r2:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=("red" if rf_mean_r2 < 0 else "black"),
        )
    ax.text(
        x[1],
        y1 + offset,
        f"p={rf_p}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    # ---- Axes & legend ----
    ax.set_xticks(x)
    ax.set_xticklabels(["EN", "RF"])
    ax.set_ylabel("Relative feature importance")  # concise label, no title

    if show_legend and bars_for_legend:
        seen = set()
        handles, labels = [], []
        for feat, handle in bars_for_legend:
            if feat not in seen:
                seen.add(feat)
                labels.append(feat)
                handles.append(handle)
        ax.legend(
            handles,
            labels,
            title="Feature Group",
            bbox_to_anchor=(1.02, 0.7),
            loc="center left",
        )
        sort_legend_alphanum(ax)

    # headroom for annotations
    ax.set_ylim(0, ymax * 1.35)

    fig.tight_layout()
    add_logo_to_figure(fig)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# ----- TWO-BAR (EN vs RF) STACKED MEAN FEATURE IMPORTANCE -----
# Requires: df_en, df_rf, STYLE_ORDER, COLOR_MAP, HATCH_MAP, DATA_DIR, outputs
# (All already present in your script.)

# 1) Fixed-width version
out_mean_fixed = os.path.join(
    DATA_DIR, "EN_RF_mean_pos_relmae_greater_2p_stacked_fixed.png"
)
stacked_mean_relmae_per_model(
    df_en_tidy=df_en,
    df_rf_tidy=df_rf,
    out_png=out_mean_fixed,
    style_order=STYLE_ORDER,
    color_map=COLOR_MAP,
    hatch_map=HATCH_MAP,
    min_rel_delta_mae=None,
    top_k=12,  # optional: keep top K features overall
    show_legend=True,
    data_dir=DATA_DIR,
    width_by_std=False,  # fixed-width segments
    rel_threshold_fraction=0.02,
)
outputs.append(out_mean_fixed)

# 2) Std-encoded widths (robust, 5–95% quantiles)
out_mean_std = os.path.join(
    DATA_DIR, "EN_RF_mean_pos_relmae_greater_2p_stacked_stdwidths.png"
)
stacked_mean_relmae_per_model(
    df_en_tidy=df_en,
    df_rf_tidy=df_rf,
    out_png=out_mean_std,
    style_order=STYLE_ORDER,
    color_map=COLOR_MAP,
    hatch_map=HATCH_MAP,
    min_rel_delta_mae=None,
    top_k=12,
    show_legend=True,
    data_dir=DATA_DIR,
    width_by_std=True,  # <<< enable robust std-based widths
    std_quantiles=(5, 95),  # robust range for width mapping
    base_width=0.65,  # used only if width_by_std=False
    rel_threshold_fraction=0.02,
)
outputs.append(out_mean_std)
