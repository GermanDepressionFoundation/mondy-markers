import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
        f"{results_dir}/models/{pseudonym}_04_rf_feature_importance_shap_summary.png",
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
        f"{results_dir}/models/{pseudonym}_05_rf_feature_importance_shap_mean_feature_contributions.png",
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
            len(d[model_key]["lower"]),
            len(d[model_key]["upper"]),
            len(d["train_mask"]),
        )

        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(d["timestamps"][:n]),
                "PHQ2": d["phq2_raw"][:n],
                "pred": d[model_key]["pred"][:n],
                "lower": d[model_key]["lower"][:n],
                "upper": d[model_key]["upper"][:n],
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

        # Fehlerband
        plt.fill_between(
            df["ts"],
            df["lower"],
            df["upper"],
            color=pred_color,
            alpha=0.25,
            label="95 %-PI",
        )

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
                "lower": d[model_key]["lower"][:n],
                "upper": d[model_key]["upper"][:n],
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
