# %% Data Loading and Setup
import json
import os
from collections import Counter

import matplotlib.pyplot as plt

from utils import PLOT_STYLES, add_logo_to_figure

plt.rcParams["font.family"] = PLOT_STYLES["font"]

import os

import forestci as fci
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# %% Utility and Pipeline Functions


def ensure_results_dir():
    os.makedirs("results/models", exist_ok=True)


def prepare_data(df, target_column):
    df = df.drop(columns=["timestamp_utc", "pseudonym", "woche_PHQ9_sum"])
    df = df.dropna(subset=[target_column])
    df = df.astype(float)
    y = df[target_column].values
    X = df.drop(columns=[target_column])
    return X, y, X.columns


def preprocess_pipeline():
    return Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    max_iter=20,
                    random_state=RANDOM_STATE,
                    estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                ),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )


def evaluate_and_plot_parity(y_true, y_pred, r2, mae, pseudonym, model_name, suffix=""):
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
        f"results/models/{pseudonym}_{suffix}_{model_name}_parity.png",
        dpi=300,
        bbox_inches=None,
    )
    plt.close()


def plot_elasticnet_coefficients(feature_names, coefficients, pseudonym):
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
        f"results/models/{pseudonym}_02_elasticnet_feature_importance.png", dpi=300
    )
    plt.close()


def plot_shap_summary_and_bar(shap_values, X_test, feature_names, pseudonym):
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    fig = plt.gcf()
    add_logo_to_figure(fig)
    plt.savefig(
        f"results/models/{pseudonym}_04_rf_feature_importance_shap_summary.png",
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
        f"results/models/{pseudonym}_05_rf_feature_importance_shap_mean_feature_contributions.png",
        dpi=300,
    )
    plt.close()


# Helper: RMSSD computation
def compute_rmssd(series):
    values = pd.Series(series).astype("float").to_numpy()
    diffs = np.diff(values)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    return np.sqrt(np.nanmean(diffs**2))


def plot_feature_importance_stats(
    elasticnet_feature_importances, rf_feature_importances
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
        "results/feature_importance_boxplots_stacked_aligned.png",
        dpi=300,
    )


def plot_mae_rmssd_bar(
    results_dict, model_key="elastic", save_path=f"results/mae_rmssd_bar.png"
):
    """
    Bar-Plot von MAE (Elasticnet oder RF) + RMSSD_PHQ2 pro Proband.

    Parameters
    ----------
    model_key : {"elastic", "rf"}
    """

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


def plot_mae_rmssd_bar_2(
    results_dict, model_key="elastic", save_path="results/mae_rmssd_bar.png"
):
    """
    Bar-Plot von MAE (Elasticnet oder RF) + RMSSD_PHQ2 pro Proband.

    Parameters
    ----------
    model_key : {"elastic", "rf"}
    """
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


def plot_phq2_timeseries_from_results(
    plot_data, model_key="rf", save_dir="results/timeseries"
):
    """
    Erstellt für jeden Probanden einen Zeitreihen-Plot:
    PHQ-2-Rohwerte + Prediction ± 95 %-Intervall.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _equalize(*arrays):
        min_len = min(map(len, arrays))
        return [arr[:min_len] for arr in arrays]

    for pseudo, d in plot_data.items():
        if model_key not in d:
            continue

        # --- DataFrame vorbereiten ---------------------------------------
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

        # --- Plot ---------------------------------------------------------
        plt.figure(figsize=(10, 5))

        # Rohdaten als Scatter
        m_train = df["is_train"]
        plt.scatter(
            df.loc[m_train, "ts"],
            df.loc[m_train, "PHQ2"],
            label="train",
            color=PLOT_STYLES["colors"]["Rohdaten_train"],
            s=22,
            alpha=0.9,
        )
        plt.scatter(
            df.loc[~m_train, "ts"],
            df.loc[~m_train, "PHQ2"],
            label="test",
            color=PLOT_STYLES["colors"]["Rohdaten_test"],
            s=22,
            alpha=0.9,
        )

        # Predicitons als Linie + 95 PI %-Intervall
        pred_color = PLOT_STYLES["colors"][
            "Elasticnet" if model_key == "elastic" else "RF"
        ]
        plt.plot(
            df["ts"],
            df["pred"],
            label=f"{model_key.upper()}-Pred",
            color=pred_color,
            lw=1.5,
        )
        plt.fill_between(
            df["ts"],
            df["lower"],
            df["upper"],
            color=pred_color,
            alpha=0.25,
            label="95 %-PI",
        )

        plt.title(f"{pseudo} – PHQ-2 Verlauf ({model_key.upper()})")
        plt.xlabel("Datum")
        plt.ylabel("PHQ-2-Score")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_dir}/{pseudo}_{model_key}.png"
        fig = plt.gcf()
        add_logo_to_figure(fig)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Info] Plot gespeichert: {out_path}")


def plot_phq2_timeseries_from_results_2(
    plot_data, model_key="rf", save_dir="results/timeseries"
):
    """
    Plot: PHQ-2-Rohdaten (Linie, getrennt nach Train/Test, Lücken bei NaN)
          + Modell-Prädiktion ± 95 %-PI.
    """
    os.makedirs(save_dir, exist_ok=True)

    for pseudo, d in plot_data.items():
        if model_key not in d:  # Modell fehlt
            continue

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

        # Train-Linie: Test-Werte als NaN -> Linienbruch
        y_train_line = df["PHQ2"].where(df["is_train"], np.nan)
        plt.plot(
            df["ts"],
            y_train_line,
            label="Rohdaten (Train)",
            color=PLOT_STYLES["colors"]["Rohdaten_train"],
            lw=1.3,
        )

        # Test-Linie: Train-Werte als NaN
        y_test_line = df["PHQ2"].where(~df["is_train"], np.nan)
        plt.plot(
            df["ts"],
            y_test_line,
            label="Rohdaten (Test)",
            color=PLOT_STYLES["colors"]["Rohdaten_test"],
            lw=1.3,
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

        plt.title(f"{pseudo} – PHQ-2 Verlauf ({model_key.upper()})")
        plt.xlabel("Datum")
        plt.ylabel("PHQ-2-Score")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_dir}/{pseudo}_{model_key}.png"
        fig = plt.gcf()
        add_logo_to_figure(fig)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Info] Plot gespeichert: {out_path}")


def plot_phq2_timeseries_from_results_3(
    plot_data, model_key="rf", save_dir="results/timeseries"
):
    """
    Plot: PHQ-2-Rohdaten (Linie + Marker, getrennt nach Train/Test),
          Modell-Prädiktion + 95 %-PI.
    """
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
            label="Rohdaten (Train)",
        )
        plt.plot(
            df["ts"],
            df["PHQ2"].where(~df["is_train"], np.nan),
            color=c_test,
            lw=1.3,
            label="Rohdaten (Test)",
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

        plt.title(f"{pseudo} – PHQ-2 Verlauf ({model_key.upper()})")
        plt.xlabel("Datum")
        plt.ylabel("PHQ-2-Score")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_dir}/{pseudo}_{model_key}.png"
        fig = plt.gcf()
        add_logo_to_figure(fig)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Info] Plot gespeichert: {out_path}")


# %% Main Processing Function


def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    plot_data = {}

    elasticnet_feature_importances = {
        feature: []
        for feature in df_raw.columns
        if feature not in ["pseudonym", target_column]
    }
    rf_feature_importances = {
        feature: []
        for feature in df_raw.columns
        if feature not in ["pseudonym", target_column]
    }
    elasticnet_feature_counts = Counter()
    rf_feature_counts = Counter()
    ensure_results_dir()

    # --- Per-participant loop ---
    for pseudonym in pseudonyms:
        print(f"Processing {pseudonym}")
        df_participant = df_raw[df_raw["pseudonym"] == pseudonym].iloc[:365]

        target_mask = df_participant[target_column].notna()
        df_model = df_participant.loc[target_mask].reset_index(drop=True)
        timestamps_model = df_model["timestamp_utc"].astype(str).tolist()
        y_full = df_model[target_column].to_numpy()

        # Compute RMSSD for PHQ-2
        rmssd_phq2 = compute_rmssd(df_participant["abend_PHQ2_sum"].values)

        # --- Prepare data ---
        X, y, feature_names = prepare_data(df_model, target_column)

        # --- Preprocessing ---
        print(f"Preprocessing data for {pseudonym}...")
        preproc = preprocess_pipeline()
        X_preproc = preproc.fit_transform(X)

        # --- Train/test split ---
        print("Splitting data into train/test sets...")
        indices = np.arange(len(X_preproc))
        X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
            X_preproc,
            y,
            indices,
            test_size=0.3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        train_mask = np.zeros(len(X_preproc), dtype=bool)
        train_mask[idx_tr] = True
        test_mask = ~train_mask

        try:
            # --- Elastic Net ---
            print("Training Elastic Net...")
            elastic = ElasticNetCV(cv=5, l1_ratio=0.5, random_state=RANDOM_STATE)
            elastic.fit(X_train, y_train)
            y_pred_elastic = elastic.predict(X_test)
            r2_elastic = round(r2_score(y_test, y_pred_elastic), 3)
            mae_elastic = round(mean_absolute_error(y_test, y_pred_elastic), 3)

            evaluate_and_plot_parity(
                y_test,
                y_pred_elastic,
                r2_elastic,
                mae_elastic,
                pseudonym,
                "elasticnet",
                "01",
            )
            plot_elasticnet_coefficients(feature_names, elastic.coef_, pseudonym)

            # Collect Elastic Net importances
            nonzero_coef_df = pd.DataFrame(
                {"Feature": feature_names, "Coefficient": elastic.coef_}
            )
            for feature, coef in zip(feature_names, elastic.coef_):
                elasticnet_feature_importances[feature].append(abs(coef))
            nonzero_coef_df["AbsCoef"] = nonzero_coef_df["Coefficient"].abs()
            top_elastic = (
                nonzero_coef_df[nonzero_coef_df["Coefficient"] != 0]
                .nlargest(TOP_K, "AbsCoef")["Feature"]
                .tolist()
            )
            elasticnet_feature_counts.update(top_elastic)
        except Exception as e:
            print(f"Elastic Net failed for {pseudonym}: {e}")
            continue

        # ---------- Elastic Net: Bootstrapped Prädiktionsintervall -------------
        print("Computing bootstrapped prediction intervals for Elastic Net...")
        n_boot = 300
        boot_preds = np.empty((n_boot, len(X_preproc)))

        for i in range(n_boot):
            boot_idx = np.random.randint(0, len(X_train), len(X_train))
            X_b, y_b = X_train[boot_idx], y_train[boot_idx]
            en_b = ElasticNetCV(cv=5, l1_ratio=0.5, random_state=RANDOM_STATE + i)
            en_b.fit(X_b, y_b)
            boot_preds[i] = en_b.predict(X_preproc)

        # Punktvorhersage = Median der Bootstrap-Verteilungen
        y_pred_elastic_full = np.median(boot_preds, axis=0)
        en_lower = np.percentile(boot_preds, 2.5, axis=0)
        en_upper = np.percentile(boot_preds, 97.5, axis=0)

        # --- Random Forest ---
        print("Training Random Forest...")
        rf = RandomForestRegressor(random_state=RANDOM_STATE)
        cv_params = {
            "max_depth": [4, 5, 6, 7, 8],
            "n_estimators": [75, 100, 125, 150, 175, 200],
        }
        rf_cv = GridSearchCV(
            rf, cv_params, scoring="neg_mean_absolute_error", cv=5, n_jobs=4
        )
        rf_cv.fit(X_train, y_train)

        best_rf = rf_cv.best_estimator_
        y_pred_rf = best_rf.predict(X_test)
        r2_rf = round(r2_score(y_test, y_pred_rf), 3)
        mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 3)

        y_pred_rf_full = best_rf.predict(X_preproc)
        # ---------- Jackknife-Varianz & PI -------------------------------------
        print("Computing jackknife variance for Random Forest...")
        rf_var = fci.random_forest_error(best_rf, X_train.shape, X_preproc)
        rf_sigma = np.sqrt(rf_var)
        rf_lower = y_pred_rf_full - 1.96 * rf_sigma
        rf_upper = y_pred_rf_full + 1.96 * rf_sigma

        evaluate_and_plot_parity(
            y_test, y_pred_rf, r2_rf, mae_rf, pseudonym, "rf", "03"
        )

        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X_test)
        plot_shap_summary_and_bar(shap_values, X_test, feature_names, pseudonym)

        mean_shap = np.abs(shap_values).mean(axis=0)
        for feature, shap_val in zip(feature_names, mean_shap):
            rf_feature_importances[feature].append(shap_val)
        top_indices = mean_shap.argsort()[::-1][:TOP_K]
        top_rf_features = [feature_names[i] for i in top_indices]
        rf_feature_counts.update(top_rf_features)

        raw_samples_not_covered_by_elastic_pi_ser = pd.Series(
            (y < en_lower) | (y > en_upper)
        )
        percent_raw_samples_not_covered_by_elastic_prediction_interval = (
            raw_samples_not_covered_by_elastic_pi_ser.value_counts().get(True, 0)
            / len(y)
            * 100
        )
        percent_raw_samples_not_covered_by_elastic_prediction_interval_train = (
            raw_samples_not_covered_by_elastic_pi_ser.loc[train_mask]
            .value_counts()
            .get(True, 0)
            / len(y)
            * 100
        )
        percent_raw_samples_not_covered_by_elastic_prediction_interval_test = (
            raw_samples_not_covered_by_elastic_pi_ser.loc[test_mask]
            .value_counts()
            .get(True, 0)
            / len(y)
            * 100
        )

        raw_samples_not_covered_by_rf_pi_ser = pd.Series(
            (y < rf_lower) | (y > rf_upper)
        )
        percent_raw_samples_not_covered_by_rf_prediction_interval = (
            raw_samples_not_covered_by_rf_pi_ser.value_counts().get(True, 0)
            / len(y)
            * 100
        )
        percent_raw_samples_not_covered_by_rf_prediction_interval_train = (
            raw_samples_not_covered_by_rf_pi_ser.loc[train_mask]
            .value_counts()
            .get(True, 0)
            / len(y)
            * 100
        )
        percent_raw_samples_not_covered_by_rf_prediction_interval_test = (
            raw_samples_not_covered_by_rf_pi_ser.loc[test_mask]
            .value_counts()
            .get(True, 0)
            / len(y)
            * 100
        )

        # Store per-participant results
        results[pseudonym] = {
            "r2_elastic": r2_elastic,
            "mae_elastic": mae_elastic,
            "r2_rf": r2_rf,
            "mae_rf": mae_rf,
            "rmssd_phq2": rmssd_phq2,
            "percent_raw_sample_not_covered_by_elastic_prediction_interval": percent_raw_samples_not_covered_by_elastic_prediction_interval,
            "percent_raw_sample_not_covered_by_elastic_prediction_interval_train": percent_raw_samples_not_covered_by_elastic_prediction_interval_train,
            "percent_raw_sample_not_covered_by_elastic_prediction_interval_test": percent_raw_samples_not_covered_by_elastic_prediction_interval_test,
            "percent_raw_sample_not_covered_by_rf_prediction_interval": percent_raw_samples_not_covered_by_rf_prediction_interval,
            "percent_raw_sample_not_covered_by_rf_prediction_interval_train": percent_raw_samples_not_covered_by_rf_prediction_interval_train,
            "percent_raw_sample_not_covered_by_rf_prediction_interval_test": percent_raw_samples_not_covered_by_rf_prediction_interval_test,
            # placeholder, will fill low_variance after we compute percentiles
            "low_variance_candidate": None,
            "elastic_mae_lower_rmssd_phq2": None,
            "rf_mae_lower_rmssd_phq2": None,
            "elastic_mae+10p_lower_rmssd_phq2": None,
            "rf_mae+10p_lower_rmssd_phq2": None,
        }
        plot_data[pseudonym] = {
            "timestamps": timestamps_model,
            "phq2_raw": y_full.tolist(),
            "train_mask": train_mask.tolist(),
            "test_mask": test_mask.tolist(),
            "elastic": {
                "pred": y_pred_elastic_full.tolist(),
                "lower": en_lower.tolist(),
                "upper": en_upper.tolist(),
            },
            "rf": {
                "pred": y_pred_rf_full.tolist(),
                "lower": rf_lower.tolist(),
                "upper": rf_upper.tolist(),
            },
        }

    # Compute 25th percentile thresholds across all participants
    phq2_thresh = np.nanpercentile(rmssd_values_phq2, 25)
    print(f"25th percentile thresholds: PHQ-2={phq2_thresh:.3f}")

    # Flag participants as low-variance candidates
    for pseudonym in results:
        is_low_var = results[pseudonym]["rmssd_phq2"] < phq2_thresh
        results[pseudonym]["low_variance_candidate"] = is_low_var

        is_elastic_mae_lower_rmssd_phq2 = (
            results[pseudonym]["mae_elastic"] < results[pseudonym]["rmssd_phq2"]
        )
        results[pseudonym][
            "elastic_mae_lower_rmssd_phq2"
        ] = is_elastic_mae_lower_rmssd_phq2

        is_rf_mae_lower_rmssd_phq2 = (
            results[pseudonym]["mae_rf"] < results[pseudonym]["rmssd_phq2"]
        )
        results[pseudonym]["rf_mae_lower_rmssd_phq2"] = is_rf_mae_lower_rmssd_phq2
        is_elastic_mae_plus_10p_lower_rmssd_phq2 = (
            results[pseudonym]["mae_elastic"] * 1.1 < results[pseudonym]["rmssd_phq2"]
        )
        results[pseudonym][
            "elastic_mae+10p_lower_rmssd_phq2"
        ] = is_elastic_mae_plus_10p_lower_rmssd_phq2
        is_rf_mae_plus_10p_lower_rmssd_phq2 = (
            results[pseudonym]["mae_rf"] * 1.1 < results[pseudonym]["rmssd_phq2"]
        )
        results[pseudonym][
            "rf_mae+10p_lower_rmssd_phq2"
        ] = is_rf_mae_plus_10p_lower_rmssd_phq2

    # Filter importances based on criteria
    valid_pseudonyms = {
        p
        for p, res in results.items()
        if res.get("elastic_mae+10p_lower_rmssd_phq2", True)
    }
    print(
        f"Valid pseudonyms after filtering: {len(valid_pseudonyms)} of {len(results)}"
    )

    filtered_elasticnet_feature_importances = {
        feature: [
            value
            for i, (pseudonym, res) in enumerate(results.items())
            if pseudonym in valid_pseudonyms
            and i < len(elasticnet_feature_importances[feature])
            for value in [elasticnet_feature_importances[feature][i]]
        ]
        for feature in elasticnet_feature_importances
    }

    filtered_rf_feature_importances = {
        feature: [
            value
            for i, (pseudonym, res) in enumerate(results.items())
            if pseudonym in valid_pseudonyms
            and i < len(rf_feature_importances[feature])
            for value in [rf_feature_importances[feature][i]]
        ]
        for feature in rf_feature_importances
    }

    # Compute global feature stats
    elasticnet_stats = {
        feature: {
            "mean": np.mean(values) if values else 0.0,
            "std": np.std(values) if values else 0.0,
        }
        for feature, values in filtered_elasticnet_feature_importances.items()
    }

    rf_stats = {
        feature: {
            "mean": np.mean(values) if values else 0.0,
            "std": np.std(values) if values else 0.0,
        }
        for feature, values in filtered_rf_feature_importances.items()
    }

    plot_feature_importance_stats(
        elasticnet_feature_importances, rf_feature_importances
    )

    return (
        results,
        elasticnet_feature_counts,
        rf_feature_counts,
        elasticnet_stats,
        rf_stats,
        plot_data,
    )


# %% Run the pipeline
# Load dataset
df_raw = pd.read_pickle("data/df_merged_v2.pickle")

# Map IDs to pseudonyms
json_file_path = "config/id_to_pseudonym.json"
# Load the mapping into a Python dictionary
with open(json_file_path, "r", encoding="utf-8") as f:
    id_to_pseudonym = json.load(f)

df_raw["pseudonym"] = df_raw["patient_id"].map(id_to_pseudonym)
df_raw = df_raw.dropna(subset=["pseudonym"])
df_raw = df_raw.drop(
    columns=[
        "patient_id",
        # "total_activity_min",
        # "total_sm_rx",
        # "total_sm_tx",
        # "total_com_rx",
        # "total_com_tx",
    ]
)
# df_raw = df_raw.drop_duplicates()

target_column = "abend_PHQ2_sum"
pseudonyms = df_raw["pseudonym"].unique()

TOP_K = 15  # max number of top features to count per participant
RANDOM_STATE = 42

results, elastic_counts, rf_counts, elastic_stats, rf_stats, plot_data = (
    process_participants(df_raw, pseudonyms, target_column)
)

# %% Plot MAE and RMSSD for each participant
plot_mae_rmssd_bar_2(
    results, model_key="elastic", save_path=f"results/mae_elastic_rmssd_bar.png"
)
plot_mae_rmssd_bar_2(results, model_key="rf", save_path=f"results/mae_rf_rmssd_bar.png")

# %% Plot PHQ-2 time series with predictions
plot_phq2_timeseries_from_results_3(plot_data, "rf")
plot_phq2_timeseries_from_results_3(plot_data, "elastic")

# %% Save evaluation metrics
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv("results/model_performance_summary.csv")

# Save feature importance counts
pd.Series(elastic_counts).sort_values(ascending=False).to_csv(
    "results/elasticnet_top_features.csv"
)
pd.Series(rf_counts).sort_values(ascending=False).to_csv(
    "results/randomforest_top_features.csv"
)

# Save feature importance mean and stds
pd.DataFrame(elastic_stats).T.sort_values("mean", ascending=False).to_csv(
    "results/elasticnet_feature_importance_stats.csv"
)
pd.DataFrame(rf_stats).T.sort_values("mean", ascending=False).to_csv(
    "results/randomforest_feature_importance_stats.csv"
)
