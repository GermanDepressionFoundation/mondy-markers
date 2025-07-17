# %% Data Loading and Setup
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
from .colormap import PLOT_STYLES

import numpy as np
import pandas as pd
import shap
from scipy import stats
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
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True PHQ-2")
    plt.ylabel("Predicted PHQ-2")
    plt.title(f"Parity Plot\nR² = {r2:.2f}, MAE = {mae:.2f} | n_test = {len(y_true)}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/models/{pseudonym}_{suffix}_{model_name}_parity.png", dpi=300)
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
    plt.savefig(
        f"results/models/{pseudonym}_02_elasticnet_feature_importance.png",
        dpi=300,
    )
    plt.close()


def plot_shap_summary_and_bar(shap_values, X_test, feature_names, pseudonym):
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
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


def plot_feature_importance_stats(elasticnet_feature_importances, rf_feature_importances):

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
    plt.savefig(
        "results/feature_importance_boxplots_stacked_aligned.png",
        dpi=300,
    )

def plot_mae_rmssd_bar(results_dict, save_path="results/mae_rmssd_bar.png"):
    """
    Stellt MAE (linke Y-Achse) und RMSSD_PHQ2 (rechte Y-Achse) pro Proband
    als gruppierte Balken dar. Beide Y-Achsen sind gleich skaliert.

    Parameters
    ----------
    results_dict : dict
        {pseudonym: {"mae_elastic": float, "rmssd_phq2": float}, …}
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pseudonyms   = list(results_dict.keys())
    mae_vals     = [results_dict[p]["mae_elastic"] for p in pseudonyms]
    rmssd_vals   = [results_dict[p]["rmssd_phq2"]  for p in pseudonyms]

    x = np.arange(len(pseudonyms))
    width = 0.4

    # Gemeinsames Y-Limit bestimmen (+1 für etwas mehr Platz)
    all_vals = mae_vals + rmssd_vals
    y_max = max(all_vals) + 1

    fig, ax1 = plt.subplots(figsize=(max(8, len(pseudonyms) * .55), 6))
    ax2 = ax1.twinx()

    ax1.bar(x - width/2, mae_vals,   width, label="MAE (Elastic Net)",
            color="tab:blue")
    ax2.bar(x + width/2, rmssd_vals, width, label="RMSSD PHQ-2",
            color="tab:orange")

    ax1.set_ylabel("MAE")
    ax2.set_ylabel("RMSSD PHQ-2")
    ax1.set_xticks(x); ax1.set_xticklabels(pseudonyms, rotation=90)
    ax1.set_title("MAE vs RMSSD (PHQ-2) pro Proband")

    # Achsen gleich skalieren
    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    # gemeinsame Legende, automatisch platzieren
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    fig.legend(h1+h2, l1+l2, loc="best")

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_phq2_timeseries_from_results(plot_data,
                                      model_key="rf",
                                      save_dir="results/timeseries"):
    """
    Erstellt für jeden Probanden einen Zeitreihen-Plot:
    PHQ-2-Rohwerte + Prediction ± 95 %-Intervall.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _equalize(*arrays):
        min_len = min(map(len, arrays))
        return [arr[:min_len] for arr in arrays]

    for pseudo, data in plot_data.items():
        if model_key not in data:
            continue                    

        # ---- Arrays vereinheitlichen --------------------------------------
        ts, phq2, pred, low, up = _equalize(
            data["timestamps"],
            data["phq2_raw"],
            data[model_key]["pred"],
            data[model_key]["lower"],
            data[model_key]["upper"],
        )

        df = pd.DataFrame({
            "ts":    pd.to_datetime(ts),
            "PHQ2":  phq2,
            "pred":  pred,
            "lower": low,
            "upper": up,
        }).sort_values("ts")

        # ---- Plot ---------------------------------------------------------
        plt.figure(figsize=(10, 5))

        plt.plot(df["ts"], df["PHQ2"], label="PHQ-2 (roh)", alpha=.7)

        pred_line, = plt.plot(
            df["ts"], df["pred"],
            label=f"{model_key.upper()}-Pred"
        )
        pred_color = pred_line.get_color()      

        plt.fill_between(
            df["ts"], df["lower"], df["upper"],
            color=pred_color, alpha=.25,           
            label="95 %-CI"
        )

        plt.title(f"{pseudo} – PHQ-2 vs. {model_key.upper()}")
        plt.xlabel("Datum")
        plt.ylabel("PHQ-2-Score")
        plt.legend()
        plt.tight_layout()

        out_path = f"{save_dir}/{pseudo}_{model_key}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[Info] Plot gespeichert: {out_path}")

# %% Main Processing Function

def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    rmssd_values_phq9 = []
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

        target_mask      = df_participant[target_column].notna()
        df_model         = df_participant.loc[target_mask].reset_index(drop=True)
        timestamps_model = df_model["timestamp_utc"].astype(str).tolist()
        y_full           = df_model[target_column].to_numpy()

        df_weekly = df_participant[["timestamp_utc", "woche_PHQ9_sum"]].copy()
        df_weekly.set_index("timestamp_utc", inplace=True)
        df_weekly = df_weekly.resample("7D").sum(min_count=1)

        # Compute RMSSD for PHQ-2 & PHQ-9
        rmssd_phq2 = compute_rmssd(df_participant["abend_PHQ2_sum"].values)
        rmssd_phq9 = compute_rmssd(df_weekly["woche_PHQ9_sum"].values)
        rmssd_values_phq2.append(rmssd_phq2)
        rmssd_values_phq9.append(rmssd_phq9)

        # --- Prepare data ---
        X, y, feature_names = prepare_data(df_model, target_column)

        # --- Preprocessing ---
        preproc = preprocess_pipeline()
        X_preproc = preproc.fit_transform(X)

        # --- Train/test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_preproc, y, test_size=0.3, shuffle=True, random_state=RANDOM_STATE
        )

        try:
            # --- Elastic Net ---
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
        
        # ---  Vorhersagen auf der *vollständigen* Zeitreihe -----------------
        y_pred_elastic_full = elastic.predict(X_preproc)
        # Unsicherheit EN: konstantes CI über Trainingsresiduals
        resid_std   = np.std(y_train - elastic.predict(X_train), ddof=1)
        z_score     = stats.norm.ppf(0.975)               # 95 %-CI
        en_margin   = z_score * resid_std
        en_lower    = y_pred_elastic_full - en_margin
        en_upper    = y_pred_elastic_full + en_margin

        # --- Random Forest ---
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

        # RF-Unsicherheit: Verteilung aller Bäume
        tree_preds  = np.vstack([t.predict(X_preproc) for t in best_rf.estimators_])
        rf_lower    = np.percentile(tree_preds,  2.5, axis=0)
        rf_upper    = np.percentile(tree_preds, 97.5, axis=0)
        y_pred_rf_full = tree_preds.mean(axis=0)

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

        # Store per-participant results
        results[pseudonym] = {
            "r2_elastic": r2_elastic,
            "mae_elastic": mae_elastic,
            "r2_rf": r2_rf,
            "mae_rf": mae_rf,
            "rmssd_phq2": rmssd_phq2,
            "rmssd_phq9": rmssd_phq9,
            # placeholder, will fill low_variance after we compute percentiles
            "low_variance_candidate": None,
        }
        plot_data[pseudonym] = {
            "timestamps": timestamps_model,
            "phq2_raw":   y_full.tolist(),
            "elastic": {
                "pred":  y_pred_elastic_full.tolist(),
                "lower": en_lower.tolist(),
                "upper": en_upper.tolist(),
            },
            "rf": {
                "pred":  y_pred_rf_full.tolist(),
                "lower": rf_lower.tolist(),
                "upper": rf_upper.tolist(),
            },
        }

    # Compute 25th percentile thresholds across all participants
    phq2_thresh = np.nanpercentile(rmssd_values_phq2, 25)
    phq9_thresh = np.nanpercentile(rmssd_values_phq9, 25)
    print(
        f"25th percentile thresholds: PHQ-2={phq2_thresh:.3f}, PHQ-9={phq9_thresh:.3f}"
    )

    # Flag participants as low-variance candidates
    for pseudonym in results:
        is_low_var = (results[pseudonym]["rmssd_phq2"] < phq2_thresh) or (
            results[pseudonym]["rmssd_phq9"] < phq9_thresh
        )
        results[pseudonym]["low_variance_candidate"] = is_low_var

    # Compute global feature stats
    elasticnet_stats = {
        feature: {
            "mean": np.mean(values) if values else 0.0,
            "std": np.std(values) if values else 0.0,
        }
        for feature, values in elasticnet_feature_importances.items()
    }
    rf_stats = {
        feature: {
            "mean": np.mean(values) if values else 0.0,
            "std": np.std(values) if values else 0.0,
        }
        for feature, values in rf_feature_importances.items()
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
df_raw = pd.read_pickle("data/df_merged_v1.pickle")

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
        "total_activity_min",
        "total_sm_rx",
        "total_sm_tx",
        "total_com_rx",
        "total_com_tx",
    ]
)
df_raw = df_raw.drop_duplicates()

target_column = "abend_PHQ2_sum"
pseudonyms = df_raw["pseudonym"].unique()

TOP_K = 15  # max number of top features to count per participant
RANDOM_STATE = 42

results, elastic_counts, rf_counts, elastic_stats, rf_stats, plot_data = process_participants(
    df_raw, pseudonyms, target_column
)

# %% Plot MAE and RMSSD for each participant
plot_mae_rmssd_bar(results)

# %% Plot PHQ-2 time series with predictions
plot_phq2_timeseries_from_results(plot_data, "rf")
plot_phq2_timeseries_from_results(plot_data, "elastic")

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
