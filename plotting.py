import os

os.environ["MPLBACKEND"] = "Agg"  # muss vor matplotlib-Import passieren
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()  # interaktiven Modus deaktivieren

import re

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import TimeSeriesSplit

from utils import PLOT_STYLES, add_logo_to_figure

plt.rcParams["font.family"] = PLOT_STYLES["font"]


def evaluate_and_plot_parity(
    results_dir, y_true, y_pred, r2, mae, pseudonym, model_name, suffix=""
):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True PHQ-2")
    plt.ylabel("Predicted PHQ-2")
    plt.title(f"Parity Plot\nR² = {r2:.2f}, MAE = {mae:.2f} | n_test = {len(y_true)}")
    plt.grid(True)
    # fig = plt.gcf()
    # add_logo_to_figure(fig)
    plt.savefig(
        f"{results_dir}/models/{pseudonym}_{suffix}_{model_name}_parity.png",
        dpi=300,
        bbox_inches=None,
    )
    plt.close()


def plot_elasticnet_coefficients(results_dir, feature_names, coefficients, pseudonym):
    coef_df = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefficients}
    ).sort_values("Coefficient", key=np.abs, ascending=False)

    plt.figure(figsize=(10, 6))
    colors = ["steelblue" if c > 0 else "salmon" for c in coef_df["Coefficient"]]
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Elastic Net Coefficient")
    plt.title("Feature Importance (Elastic Net Coefficients)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig = plt.gcf()
    add_logo_to_figure(fig)
    plt.savefig(
        f"{results_dir}/models/{pseudonym}_02_elasticnet_feature_importance.png",
        dpi=300,
    )
    plt.close()


def plot_shap_summary_and_bar(
    results_dir, shap_values, X_test, feature_names, pseudonym
):
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    fig = plt.gcf()
    add_logo_to_figure(fig)
    plt.savefig(
        f"{results_dir}/models/{pseudonym}_04_feature_importance_shap_summary.png",
        dpi=300,
    )
    plt.close()

    mean_shap = np.mean(shap_values, axis=0)
    shap_df = pd.DataFrame(
        {"Feature": feature_names, "MeanSHAP": mean_shap}
    ).sort_values("MeanSHAP", key=abs, ascending=False)

    plt.figure(figsize=(10, 6))
    colors = ["steelblue" if v > 0 else "salmon" for v in shap_df["MeanSHAP"]]
    plt.barh(shap_df["Feature"], shap_df["MeanSHAP"], color=colors)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Mean SHAP Value (Impact on Prediction)")
    plt.title("SHAP Feature Contributions (Centered at 0)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig = plt.gcf()
    add_logo_to_figure(fig)
    plt.savefig(
        f"{results_dir}/models/{pseudonym}_05_feature_importance_shap_mean_feature_contributions.png",
        dpi=300,
    )
    plt.close()


def plot_feature_importance_stats(
    results_dir, elasticnet_feature_importances, rf_feature_importances
):

    all_features = sorted(elasticnet_feature_importances.keys())

    # Prepare raw distributions for aligned order
    elastic_data = [elasticnet_feature_importances[f] for f in all_features]
    rf_data = [rf_feature_importances[f] for f in all_features]
    # --- Create stacked subplots ---
    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, len(all_features) * 0.4), 14), sharey=False
    )

    # ===== Elastic Net subplot =====
    axes[0].boxplot(
        elastic_data,
        labels=all_features,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue", alpha=0.6),
    )
    # Overlay participant-level points
    for i, feat in enumerate(all_features, start=1):
        vals = np.array(elasticnet_feature_importances[feat])
        jitter_x = np.random.normal(i, 0.05, size=len(vals))
        axes[0].scatter(jitter_x, vals, color="black", alpha=0.5, s=10)

    axes[0].set_title("Elastic Net Feature Importances (per-participant distributions)")
    axes[0].set_ylabel("Importance (|coef|)")
    axes[0].tick_params(axis="x", rotation=90)

    # ===== Random Forest subplot =====
    axes[1].boxplot(
        rf_data,
        labels=all_features,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", alpha=0.6),
    )
    # Overlay participant-level points
    for i, feat in enumerate(all_features, start=1):
        vals = np.array(rf_feature_importances[feat])
        jitter_x = np.random.normal(i, 0.05, size=len(vals))
        axes[1].scatter(jitter_x, vals, color="black", alpha=0.5, s=10)

    axes[1].set_title(
        "Random Forest Feature Importances (per-participant distributions)"
    )
    axes[1].set_ylabel("Importance (mean |SHAP|)")
    axes[1].tick_params(axis="x", rotation=90)

    plt.suptitle(
        "Aligned Feature Importance Distributions Across Participants", fontsize=16
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig = plt.gcf()
    add_logo_to_figure(fig)
    plt.savefig(
        f"{results_dir}/feature_importance_boxplots_stacked_aligned.png",
        dpi=300,
    )


def plot_mae_rmssd_bar(results_dir, results_dict, model_key="elastic"):
    """
    Bar-Plot von MAE (Elasticnet oder RF) + RMSSD_PHQ2 pro Proband.

    Parameters
    ----------
    model_key : {"elastic", "rf"}
    """

    save_path = f"{results_dir}/mae_rmssd_bar.png"
    pseudonyms = list(results_dict.keys())
    mae_vals = [
        results_dict[p][f"mae_{'elastic' if model_key=='elastic' else 'rf'}"]
        for p in pseudonyms
    ]
    rmssd_vals = [results_dict[p]["rmssd_phq2"] for p in pseudonyms]

    x = np.arange(len(pseudonyms))
    width = 0.4
    fig, ax1 = plt.subplots(figsize=(max(8, 0.6 * len(pseudonyms)), 5))
    ax2 = ax1.twinx()

    col_mae = PLOT_STYLES["colors"]["Elasticnet" if model_key == "elastic" else "RF"]
    col_rmssd = PLOT_STYLES["colors"]["Rohdaten_train"]

    ax1.bar(
        x - width / 2,
        mae_vals,
        width,
        label=f"MAE ({model_key.upper()})",
        color=col_mae,
    )
    ax2.bar(x + width / 2, rmssd_vals, width, label="RMSSD PHQ-2", color=col_rmssd)

    ax1.set_ylabel("MAE")
    ax2.set_ylabel("RMSSD PHQ-2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pseudonyms, rotation=90)
    ax1.set_title(f"MAE ({model_key.upper()}) vs. RMSSD pro Proband")

    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    add_logo_to_figure(fig)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mae_rmssd_bar_2(results_dir, results_dict, model_key="elastic"):
    """
    Bar-Plot von MAE (Elasticnet oder RF) + RMSSD_PHQ2 pro Proband.

    Parameters
    ----------
    model_key : {"elastic", "rf"}
    """

    save_path = f"{results_dir}/mae_rmssd_bar.png"

    pseudonyms = list(results_dict.keys())
    mae_vals = [
        results_dict[p][f"mae_{'elastic' if model_key=='elastic' else 'rf'}"]
        for p in pseudonyms
    ]
    rmssd_vals = [results_dict[p]["rmssd_phq2"] for p in pseudonyms]

    y_max = max(max(mae_vals), max(rmssd_vals)) * 1.05

    x, width = np.arange(len(pseudonyms)), 0.4
    fig, ax1 = plt.subplots(figsize=(max(8, 0.6 * len(pseudonyms)), 5))
    ax2 = ax1.twinx()

    col_mae = PLOT_STYLES["colors"]["Elasticnet" if model_key == "elastic" else "RF"]
    col_rmssd = PLOT_STYLES["colors"]["Rohdaten_train"]

    bar1 = ax1.bar(
        x - width / 2,
        mae_vals,
        width,
        label=f"MAE ({model_key.upper()})",
        color=col_mae,
    )
    bar2 = ax2.bar(
        x + width / 2, rmssd_vals, width, label="RMSSD PHQ-2", color=col_rmssd
    )

    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    ax1.set_ylabel("MAE")
    ax2.set_ylabel("RMSSD PHQ-2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pseudonyms, rotation=90)
    ax1.set_title(f"MAE ({model_key.upper()}) vs. RMSSD pro Proband")

    # -------- Legende unterhalb platzieren ------------------------------
    fig.legend(
        [bar1, bar2],
        [bar1.get_label(), bar2.get_label()],
        # loc="lower left",
        # bbox_to_anchor=(0.0, -0.12),  # x-Offset 0, y-Offset −0.12
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    add_logo_to_figure(fig)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_phq2_timeseries_from_results(results_dir, plot_data, model_key="rf"):
    """
    Plot: PHQ-2-Rohdaten (Linie + Marker, getrennt nach Train/Test),
          Modell-Prädiktion + 95 %-PI.
    """

    save_dir = f"{results_dir}/timeseries"
    os.makedirs(save_dir, exist_ok=True)

    for pseudo, d in plot_data.items():
        if model_key not in d:
            continue  # Modell fehlt

        # -------- Länge angleichen ---------------------------------------
        n = min(
            len(d["timestamps"]),
            len(d["phq2_raw"]),
            len(d[model_key]["pred"]),
            # len(d[model_key]["lower"]),
            # len(d[model_key]["upper"]),
            len(d["train_mask"]),
        )

        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(d["timestamps"][:n]),
                "PHQ2": d["phq2_raw"][:n],
                "pred": d[model_key]["pred"][:n],
                # "lower": d[model_key]["lower"][:n],
                # "upper": d[model_key]["upper"][:n],
                "is_train": np.array(d["train_mask"][:n], dtype=bool),
            }
        ).sort_values("ts")

        # -------- Plot ---------------------------------------------------
        plt.figure(figsize=(10, 5))

        # Farben & Marker
        c_train = PLOT_STYLES["colors"]["Rohdaten_train"]
        c_test = PLOT_STYLES["colors"]["Rohdaten_test"]
        m_train = "o"  # Kreismarker
        m_test = "s"  # Quadratmarker

        # Linien (NaNs sorgen für Lücken)
        plt.plot(
            df["ts"],
            df["PHQ2"].where(df["is_train"], np.nan),
            color=c_train,
            lw=1.3,
            label="Raw data (Train)",
        )
        plt.plot(
            df["ts"],
            df["PHQ2"].where(~df["is_train"], np.nan),
            color=c_test,
            lw=1.3,
            label="Raw data (Test)",
        )

        # Marker darüberstreuen
        plt.scatter(
            df.loc[df["is_train"], "ts"],
            df.loc[df["is_train"], "PHQ2"],
            color=c_train,
            marker=m_train,
            s=28,
        )
        plt.scatter(
            df.loc[~df["is_train"], "ts"],
            df.loc[~df["is_train"], "PHQ2"],
            color=c_test,
            marker=m_test,
            s=28,
        )

        # Modell-Vorhersage
        pred_color = (
            PLOT_STYLES["colors"]["Elasticnet"]
            if model_key == "elastic"
            else PLOT_STYLES["colors"]["RF"]
        )
        plt.plot(
            df["ts"],
            df["pred"],
            label=f"{model_key.upper()}-Pred",
            color=pred_color,
            lw=1.6,
        )

        # # Fehlerband
        # plt.fill_between(
        #     df["ts"],
        #     df["lower"],
        #     df["upper"],
        #     color=pred_color,
        #     alpha=0.25,
        #     label="95 %-PI",
        # )

        # Fixed Y-axis range
        plt.ylim(0, 21)

        plt.title(f"{pseudo} – PHQ-2 Trend ({model_key.upper()})")
        plt.xlabel("Date")
        plt.ylabel("PHQ-2-Score")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_dir}/{pseudo}_{model_key}.png"
        fig = plt.gcf()
        add_logo_to_figure(fig)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Info] Plot gespeichert: {out_path}")


def plot_phq2_timeseries_with_adherence_from_results(
    results_dir, plot_data, model_key="rf"
):
    """
    Create three vertically stacked subplots per participant:
      (1) PHQ-2 raw data (train/test) + model predictions
      (2) Individual adherence trend
      (3) Global adherence trend

    Adherence series are auto-interpreted as percentages if values are in [0, 1].
    Saves one PNG per participant into <results_dir>/timeseries/.
    """

    save_dir = f"{results_dir}/timeseries"
    os.makedirs(save_dir, exist_ok=True)

    for pseudo, d in plot_data.items():
        # Require model and both adherence series
        if (
            model_key not in d
            or "individual_adherence" not in d
            or "global_adherence" not in d
        ):
            continue

        # -------- align lengths safely -----------------------------------
        n = min(
            len(d["timestamps"]),
            len(d["phq2_raw"]),
            len(d[model_key]["pred"]),
            len(d["train_mask"]),
            len(d["individual_adherence"]),
            len(d["global_adherence"]),
        )
        if n == 0:
            continue

        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(d["timestamps"][:n]),
                "PHQ2": d["phq2_raw"][:n],
                "pred": d[model_key]["pred"][:n],
                "ind_adherence": d["individual_adherence"][:n],
                "glob_adherence": d["global_adherence"][:n],
                "is_train": np.array(d["train_mask"][:n], dtype=bool),
            }
        ).sort_values("ts")

        # --- colors & markers --------------------------------------------
        c_train = PLOT_STYLES["colors"]["Rohdaten_train"]
        c_test = PLOT_STYLES["colors"]["Rohdaten_test"]
        pred_color = (
            PLOT_STYLES["colors"]["Elasticnet"]
            if model_key == "elastic"
            else PLOT_STYLES["colors"]["RF"]
        )
        # allow separate colors; provide fallbacks if not defined
        ind_color = PLOT_STYLES["colors"].get("AdherenceIndividual", "#6A5ACD")
        glob_color = PLOT_STYLES["colors"].get("AdherenceGlobal", "#2CA02C")

        m_train, m_test = "o", "s"

        # --- figure with three stacked axes -------------------------------
        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(10, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.0, 1.0], "hspace": 0.06},
        )

        # ===============================================================
        #  (1) TOP: PHQ-2 raw data and model prediction
        # ===============================================================
        ax1.plot(
            df["ts"],
            df["PHQ2"].where(df["is_train"], np.nan),
            color=c_train,
            lw=1.3,
            label="Raw data (Train)",
        )
        ax1.plot(
            df["ts"],
            df["PHQ2"].where(~df["is_train"], np.nan),
            color=c_test,
            lw=1.3,
            label="Raw data (Test)",
        )

        # markers
        ax1.scatter(
            df.loc[df["is_train"], "ts"],
            df.loc[df["is_train"], "PHQ2"],
            color=c_train,
            marker=m_train,
            s=28,
        )
        ax1.scatter(
            df.loc[~df["is_train"], "ts"],
            df.loc[~df["is_train"], "PHQ2"],
            color=c_test,
            marker=m_test,
            s=28,
        )

        # prediction line
        ax1.plot(
            df["ts"],
            df["pred"],
            color=pred_color,
            lw=1.6,
            label=f"{model_key.upper()}-Pred",
        )

        ax1.set_ylim(0, 21)
        ax1.set_ylabel("PHQ-2 score")
        ax1.legend(loc="upper left")
        ax1.set_title(f"{pseudo} – PHQ-2 Trend ({model_key.upper()})")

        # ===============================================================
        #  (2) MIDDLE: Individual adherence
        # ===============================================================
        ind = pd.to_numeric(df["ind_adherence"], errors="coerce")
        if np.nanmax(ind) <= 1.0:
            ind_plot = ind * 100.0
            ind_label = "Individual adherence (%)"
            ind_ylim = (0, 100)
        else:
            ind_plot = ind
            ind_label = "Individual adherence"
            ind_ylim = None

        ax2.plot(df["ts"], ind_plot, color=ind_color, lw=1.2, label="Individual")
        ax2.fill_between(df["ts"], ind_plot, step=None, alpha=0.10, color=ind_color)
        if ind_ylim is not None:
            ax2.set_ylim(*ind_ylim)
        ax2.set_ylabel(ind_label)
        ax2.legend(loc="upper left")

        # ===============================================================
        #  (3) BOTTOM: Global adherence
        # ===============================================================
        glob = pd.to_numeric(df["glob_adherence"], errors="coerce")
        if np.nanmax(glob) <= 1.0:
            glob_plot = glob * 100.0
            glob_label = "Global adherence (%)"
            glob_ylim = (0, 100)
        else:
            glob_plot = glob
            glob_label = "Global adherence"
            glob_ylim = None

        ax3.plot(df["ts"], glob_plot, color=glob_color, lw=1.2, label="Global")
        ax3.fill_between(df["ts"], glob_plot, step=None, alpha=0.10, color=glob_color)
        if glob_ylim is not None:
            ax3.set_ylim(*glob_ylim)
        ax3.set_ylabel(glob_label)
        ax3.set_xlabel("Date")
        ax3.legend(loc="upper left")

        # ===============================================================
        #  Final layout and export
        # ===============================================================
        for ax in (ax1, ax2, ax3):
            ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        out_path = f"{save_dir}/{pseudo}_{model_key}_stacked3.png"
        add_logo_to_figure(fig)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Info] Plot gespeichert: {out_path}")


def plot_phq2_test_errors_from_results(
    results_dir, plot_data, model_key="rf", show_pred_ci=True
):
    """
    Plots PHQ-2 predictions vs. ground truth for test samples only,
    including error bars (prediction intervals) and optionally prediction errors.
    """

    save_dir = (
        f"{results_dir}/test_errors_{model_key}_no_ci"
        if not show_pred_ci
        else f"{results_dir}/test_errors_{model_key}"
    )
    os.makedirs(save_dir, exist_ok=True)

    for pseudo, d in plot_data.items():
        if model_key not in d:
            continue

        n = min(len(d["timestamps"]), len(d["phq2_raw"]), len(d[model_key]["pred"]))
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(d["timestamps"][:n]),
                "PHQ2": d["phq2_raw"][:n],
                "pred": d[model_key]["pred"][:n],
                # "lower": d[model_key]["lower"][:n],
                # "upper": d[model_key]["upper"][:n],
                "is_train": np.array(d["train_mask"][:n], dtype=bool),
            }
        ).sort_values("ts")

        df_test = df[~df["is_train"]]

        if df_test.empty:
            continue

        # Plot
        plt.figure(figsize=(10, 5))

        # Ground truth
        plt.scatter(
            df_test["ts"],
            df_test["PHQ2"],
            label="Ground truth (test)",
            color=PLOT_STYLES["colors"]["Rohdaten_test"],
            s=25,
            alpha=0.9,
        )
        # Ground truth trend line
        plt.plot(
            df_test["ts"],
            df_test["PHQ2"],
            color=PLOT_STYLES["colors"]["Rohdaten_test"],
            alpha=0.6,
            linewidth=1,
        )
        # Prediction trend line
        plt.plot(
            df_test["ts"],
            df_test["pred"],
            color=PLOT_STYLES["colors"][
                "Elasticnet" if model_key == "elastic" else "RF"
            ],
            alpha=0.6,
            linewidth=1,
        )

        if show_pred_ci:
            # Predictions with error bars
            plt.errorbar(
                df_test["ts"],
                df_test["pred"],
                yerr=[
                    df_test["pred"] - df_test["lower"],
                    df_test["upper"] - df_test["pred"],
                ],
                fmt="o",
                color=PLOT_STYLES["colors"][
                    "Elasticnet" if model_key == "elastic" else "RF"
                ],
                ecolor=PLOT_STYLES["colors"][
                    "Elasticnet" if model_key == "elastic" else "RF"
                ],
                capsize=3,
                label=f"{model_key.upper()} prediction ±95% PI",
            )
        else:
            # Ground truth
            plt.scatter(
                df_test["ts"],
                df_test["pred"],
                label="prediction",
                color=PLOT_STYLES["colors"][
                    "Elasticnet" if model_key == "elastic" else "RF"
                ],
                s=25,
                alpha=0.9,
            )
        # Optional: draw error lines
        plt.vlines(
            df_test["ts"],
            df_test["PHQ2"],
            df_test["pred"],
            color="gray",
            linestyle="dotted",
            alpha=0.6,
            label="Prediction error",
        )

        # Fixed Y-axis range
        plt.ylim(0, 21)

        plt.title(f"{pseudo} – PHQ-2 Test Predictions ({model_key.upper()})")
        plt.xlabel("Date")
        plt.ylabel("PHQ-2-Score")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_dir}/{pseudo}_{model_key}_testonly.png"
        fig = plt.gcf()
        add_logo_to_figure(fig)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Info] Test-only plot saved: {out_path}")


def tss_indices(n_samples: int, n_splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, te) in enumerate(tscv.split(np.arange(n_samples)), start=1):
        yield fold, tr, te


def plot_tss(n_samples: int = 365, n_splits: int = 5, gap: int = 0):
    """
    Visualisiert TimeSeriesSplit: Train (blau), Embargo/GAP (grau, optional), Test (orange).
    GAP wird nur im Plot gezeigt – sklearn setzt ihn nicht automatisch um.
    """
    fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * n_splits))
    y = 0
    for fold, tr, te in tss_indices(n_samples, n_splits):
        # optionaler GAP/Embargo (nur Visualisierung)
        if gap > 0:
            te_start = te.min()
            tr = tr[tr < (te_start - gap)]
            gap_idx = np.arange(te_start - gap, te_start)
        else:
            gap_idx = np.array([], dtype=int)

        ax.plot(
            tr,
            np.full_like(tr, y),
            "|",
            markersize=10,
            label="Train" if fold == 1 else "",
            color="tab:blue",
        )
        if gap_idx.size > 0:
            ax.plot(
                gap_idx,
                np.full_like(gap_idx, y),
                "|",
                markersize=10,
                label="GAP" if fold == 1 else "",
                color="0.7",
            )
        ax.plot(
            te,
            np.full_like(te, y),
            "|",
            markersize=10,
            label="Test" if fold == 1 else "",
            color="tab:orange",
        )
        ax.text(n_samples + 3, y, f"Fold {fold}", va="center", fontsize=9)
        y += 1

    ax.set_ylim(-1, y)
    ax.set_xlim(-2, n_samples + 20)
    ax.set_yticks([])
    ax.set_xlabel("Tag")
    ax.legend(loc="upper left", ncols=3)
    ax.set_title(f"TimeSeriesSplit (n_splits={n_splits}, GAP={gap})")
    plt.tight_layout()
    plt.show()


def plot_fold_metrics(df_metrics, out_dir, pseudonym, model_label):
    # x-Achse: Datum (falls vorhanden), sonst Fold-Nummer
    if df_metrics["fold_end_ts"].notna().any():
        x = pd.to_datetime(df_metrics["fold_end_ts"])
        xlab = "Fold-Ende (Datum)"
    else:
        x = df_metrics["fold"]
        xlab = "Fold"

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, df_metrics["r2"], marker="o", label=f"{model_label} R²")
    plt.plot(
        x,
        df_metrics["r2_baseline_last"],
        marker="x",
        linestyle="--",
        label="Baseline (last) R²",
    )
    plt.plot(
        x,
        df_metrics["r2_baseline_mean"],
        marker="x",
        linestyle=":",
        label="Baseline (mean) R²",
    )
    plt.axhline(0.0, color="gray", linewidth=1)
    plt.title(f"{pseudonym}: R² über Folds – {model_label}")
    plt.xlabel(xlab)
    plt.ylabel("R²")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(out_dir, f"{pseudonym}_{model_label}_fold_r2.png")
    plt.savefig(fn, dpi=150)
    plt.close()


def plot_timeseries_with_folds(
    timestamps, y, splits, metrics_df, out_path, title="PHQ-2 mit Test-Folds"
):
    """
    Zeichnet die PHQ-2 Zeitreihe, markiert jedes Testfenster (Fold) als Bereich
    und schreibt R²/MAE aus metrics_df in die Mitte des Testbereichs.
    """
    ts = pd.to_datetime(pd.Series(timestamps))  # robust gegen string/datetime
    y = pd.Series(y).astype(float).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(ts, y, lw=1.5, label="PHQ-2")

    # y-Position für Annotationen (oben im Plot)
    y_top = np.nanmax(y)
    pad = (np.nanmax(y) - np.nanmin(y)) * 0.05 if np.isfinite(y_top) else 1.0
    y_annot = y_top + pad * 0.2 if np.isfinite(y_top) else 1.0

    for fold_id, (tr, te) in enumerate(splits, 1):
        if len(te) == 0:
            continue
        start_ts = ts.iloc[te[0]]
        end_ts = ts.iloc[te[-1]]
        mid_ts = start_ts + (end_ts - start_ts) / 2

        # Testbereich einfärben
        ax.axvspan(start_ts, end_ts, color="tab:orange", alpha=0.18)

        # Metriken aus df
        row = metrics_df.loc[metrics_df["fold"] == fold_id]
        if not row.empty:
            r2 = row["r2"].values[0]
            mae = row["mae"].values[0]
            ax.text(
                mid_ts,
                y_annot,
                f"Fold {fold_id}\nR²={r2:.2f}, MAE={mae:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            )

    ax.set_title(title)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("PHQ-2")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    # >>> NEW: Balkenplots für Summaries (SHAP/PI)


def plot_importance_bars(
    summary_df: pd.DataFrame, value_col: str, title: str, outpath: str, top_k: int = 15
):
    if summary_df is None or summary_df.empty:
        return
    dfp = summary_df.sort_values(value_col, ascending=False).head(top_k)
    plt.figure(figsize=(8, max(3, 0.35 * len(dfp))))
    plt.barh(dfp["feature"][::-1], dfp[value_col][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_folds_on_timeseries(
    timestamps: pd.Series,
    y_values: np.ndarray,
    fold_metrics_df: pd.DataFrame,
    out_path: str,
    title: str = "Rolling test windows with metrics",
):
    """
    Visualizes PHQ-2 with shaded test windows.
    Expects time-based columns in fold_metrics_df:
      - test_from_ts, test_to_ts (prefers these)
      - optionally train_from_ts, train_to_ts (not used for shading, only available for context)
      - r2, mae, n_test
    Falls back to legacy index-based fields (test_start_idx/test_end_idx) if timestamps are absent.
    """
    if fold_metrics_df is None or fold_metrics_df.empty:
        return

    # Ensure timestamps are datetime-like
    ts_main = pd.to_datetime(timestamps)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_main, y_values, lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("PHQ-2")

    # Vertical extent for labels
    ymin, ymax = np.nanmin(y_values), np.nanmax(y_values)
    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        ymin, ymax = 0.0, 1.0
    yr = max(ymax - ymin, 1.0)
    label_y = ymax + 0.05 * yr

    # Work on a copy with normalized dtypes
    fm = fold_metrics_df.copy()

    # Prefer time-based columns when available
    has_time_cols = {"test_from_ts", "test_to_ts"}.issubset(set(fm.columns))
    if has_time_cols:
        fm["test_from_ts"] = pd.to_datetime(fm["test_from_ts"], errors="coerce")
        fm["test_to_ts"] = pd.to_datetime(fm["test_to_ts"], errors="coerce")

    for _, row in fm.iterrows():
        # Determine span
        if (
            has_time_cols
            and pd.notna(row.get("test_from_ts"))
            and pd.notna(row.get("test_to_ts"))
        ):
            t_start = row["test_from_ts"]
            t_end = row["test_to_ts"]
        else:
            # Fallback to legacy index-based fields
            ts_idx = int(row.get("test_start_idx", -1))
            te_idx = int(row.get("test_end_idx", -1))
            if (
                ts_idx < 0
                or te_idx < 0
                or ts_idx >= len(ts_main)
                or te_idx >= len(ts_main)
            ):
                continue
            t_start = ts_main.iloc[ts_idx]
            t_end = ts_main.iloc[te_idx]

        # Guard invalid order
        if pd.isna(t_start) or pd.isna(t_end) or t_end <= t_start:
            continue

        # Shade test window
        ax.axvspan(t_start, t_end, color="orange", alpha=0.18)

        # Label at the center of the window
        try:
            xm = t_start + (t_end - t_start) / 2
        except Exception:
            xm = t_start

        # Build annotation text
        r2 = row.get("r2", np.nan)
        mae = row.get("mae", np.nan)
        n_te = row.get("n_test", np.nan)

        parts = []
        if pd.notna(r2):
            parts.append(f"R²={float(r2):.2f}")
        if pd.notna(mae):
            parts.append(f"MAE={float(mae):.2f}")
        if pd.notna(n_te):
            parts.append(f"n={int(n_te)}")

        label = ", ".join(parts) if parts else ""

        if label:
            ax.text(
                xm,
                label_y,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.margins(x=0.02)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def safe_filename(base: str) -> str:
    # replace any character not alphanumeric, dash, underscore, or dot
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def plot_model_vs_dummyregressor(
    model_type, per_fold_df, boot_results, results_dir, filename_prefix
):
    """
    Creates a 2x2 panel:
      - ax1: R² per fold (Model vs Dummy)
      - ax2: MAE per fold (Model vs Dummy)
      - ax3: ΔR² per fold (Model - Dummy)
      - ax4: ΔMAE per fold (Dummy - Model)  [positive = better]
    Ensures identical x-axis tick positions/limits across all subplots,
    and applies colors from PLOT_STYLES for EN/RF.
    """
    # Normalize model_type and fetch colors
    mtype = str(model_type).strip().lower()
    if "rf" in mtype or mtype == "randomforest":
        model_label = "RF"
        model_color = PLOT_STYLES["colors"].get("RF", "#1f77b4")
    else:
        # treat everything else as ElasticNet
        model_label = "EN"
        model_color = PLOT_STYLES["colors"].get("Elasticnet", "#0072B2")
    dummy_color = PLOT_STYLES["colors"].get("Dummy", "#808080")

    # Make sure folds are sorted and unique
    df = per_fold_df.copy()
    df = df.sort_values("fold")
    x = df["fold"].to_numpy()

    # Layout
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    # --- ax1: R² per fold ---
    ax1.plot(x, df["r2_en"], marker="o", label=model_label, color=model_color)
    ax1.plot(x, df["r2_dummy"], marker="s", label="Dummy", color=dummy_color)
    ax1.axhline(0, color="gray", lw=1, linestyle="--")
    ax1.set_title("R² per Fold")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("R²")
    ax1.grid(True, alpha=0.3)

    # --- ax2: MAE per fold ---
    ax2.plot(x, df["mae_en"], marker="o", label=model_label, color=model_color)
    ax2.plot(x, df["mae_dummy"], marker="s", label="Dummy", color=dummy_color)
    ax2.set_title("MAE per Fold")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Mean Absolute Error")
    ax2.grid(True, alpha=0.3)

    # --- ax3: ΔR² per fold ---
    # Use barplot but strictly lock ticks/limits to the same x
    sns.barplot(
        x=x, y=df["delta_r2"], color=model_color, ax=ax3, ci=None, estimator=np.mean
    )
    ax3.axhline(0, color="gray", lw=1)
    ax3.set_title(f"ΔR² = R²({model_label}) - R²(Dummy)")
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("R² Improvement")

    # --- ax4: ΔMAE per fold ---
    sns.barplot(
        x=x, y=df["delta_mae"], color=model_color, ax=ax4, ci=None, estimator=np.mean
    )
    ax4.axhline(0, color="gray", lw=1)
    ax4.set_title(f"ΔMAE = MAE(Dummy) - MAE({model_label})")
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("MAE Improvement (positive = better)")

    # --- enforce identical x-axis ticks and limits across all axes ---
    def _sync_xaxis(ax_list, x_values):
        # Determine tick positions every 5 folds
        xmin, xmax = np.min(x_values), np.max(x_values)
        tick_step = 5
        ticks = np.arange(xmin, xmax + 1, tick_step)
        if ticks[-1] != xmax:
            # ensure the last fold is included if not perfectly divisible
            ticks = np.append(ticks, xmax)

        # Set identical ticks and labels
        for ax in ax_list:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(t)) for t in ticks])

        # Add small padding around fold range
        pad = 0.5 if len(x_values) > 1 else 0.5
        for ax in ax_list:
            ax.set_xlim(xmin - pad, xmax + pad)

    _sync_xaxis([ax1, ax2, ax3, ax4], x)

    # Legends (top row only; bottom inherits)
    handles1, labels1 = ax1.get_legend_handles_labels()
    if handles1:
        # ax1.legend(handles1, labels1, loc="best")
        pass

    # Bootstrap summary in filename footer
    r2_res = boot_results.get("R2", {})
    mae_res = boot_results.get("MAE", {})
    r2_md = r2_res.get("mean_diff", np.nan)
    r2_lo = r2_res.get("ci_low", np.nan)
    r2_hi = r2_res.get("ci_high", np.nan)
    r2_pv = r2_res.get("p_value", np.nan)

    mae_md = mae_res.get("mean_diff", np.nan)
    mae_lo = mae_res.get("ci_low", np.nan)
    mae_hi = mae_res.get("ci_high", np.nan)
    mae_pv = mae_res.get("p_value", np.nan)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    add_logo_to_figure(fig)
    summary = (
        f"Mean_dR2_{r2_md:.3f}_CI_{r2_lo:.3f}-{r2_hi:.3f}_p_{r2_pv:.4f}"
        f"_Mean_dMAE_{mae_md:.3f}_CI_{mae_lo:.3f}-{mae_hi:.3f}_p_{mae_pv:.4f}"
    )
    fname = safe_filename(f"{filename_prefix}_{summary}.png")
    plt.savefig(os.path.join(results_dir, fname), dpi=300)
    plt.close(fig)
