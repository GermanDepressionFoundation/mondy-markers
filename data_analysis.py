# %% Data Loading and Setup
import json
import os
from collections import Counter

RESULTS_DIR = "results"

import os

import forestci as fci
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from plotting import (
    evaluate_and_plot_parity,
    plot_elasticnet_coefficients,
    plot_feature_importance_stats,
    plot_mae_rmssd_bar_2,
    plot_phq2_test_errors_from_results,
    plot_phq2_timeseries_from_results,
    plot_shap_summary_and_bar,
)

# %% Utility and Pipeline Functions


def ensure_results_dir():
    os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)


def prepare_data(df, target_column):
    df = df.drop(columns=["pseudonym", "timestamp_utc", "woche_PHQ9_sum"])
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
                    # estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                ),
                # KNNImputer(),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )


# Helper: RMSSD computation
def compute_rmssd(series):
    values = pd.Series(series).astype("float").to_numpy()
    diffs = np.diff(values)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    return np.sqrt(np.nanmean(diffs**2))


# %% Main Processing Function


def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    plot_data = {}

    df_raw["day_of_week"] = df_raw["timestamp_utc"].dt.weekday
    df_raw["month_of_year"] = df_raw["timestamp_utc"].dt.month

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
        # Compute RMSSD for PHQ-2
        rmssd_phq2 = compute_rmssd(df_participant["abend_PHQ2_sum"].values)

        # --- Prepare data ---
        X, y, feature_names = prepare_data(df_participant, target_column)
        indices = np.arange(len(X))

        # --- Train/test split ---
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            indices,
            test_size=0.3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        pipeline = preprocess_pipeline()
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        X_transformed = pipeline.transform(X)

        df = df_participant.reset_index(drop=True)
        full_mask = df[target_column].notna()
        train_mask = df.loc[idx_train, target_column].notna()
        test_mask = df.loc[idx_test, target_column].notna()
        full_train_mask = df.index.isin(idx_train) & full_mask
        full_test_mask = df.index.isin(idx_test) & full_mask

        X_train = X_train_transformed[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test_transformed[test_mask]
        y_test = y_test[test_mask]
        X_traintest = X_transformed[full_mask]
        y_traintest = y[full_mask]

        full_mask = df_participant[target_column].notna()
        df_model = df_participant.loc[full_mask]
        timestamps_model = df_model["timestamp_utc"].astype(str).tolist()

        try:
            # --- Elastic Net ---
            print("Training Elastic Net...")
            elastic = ElasticNetCV(cv=5, l1_ratio=0.5, random_state=RANDOM_STATE)
            elastic.fit(X_train, y_train)
            y_pred_elastic = elastic.predict(X_test)
            r2_elastic = round(r2_score(y_test, y_pred_elastic), 2)
            mae_elastic = round(mean_absolute_error(y_test, y_pred_elastic), 1)

            evaluate_and_plot_parity(
                RESULTS_DIR,
                y_test,
                y_pred_elastic,
                r2_elastic,
                mae_elastic,
                pseudonym,
                "elasticnet",
                "01",
            )
            plot_elasticnet_coefficients(
                RESULTS_DIR, feature_names, elastic.coef_, pseudonym
            )

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
        boot_preds = np.empty((n_boot, len(X_traintest)))

        for i in range(n_boot):
            boot_idx = np.random.randint(0, len(X_train), len(X_train))
            X_b, y_b = X_train[boot_idx], y_train[boot_idx]
            en_b = ElasticNetCV(cv=5, l1_ratio=0.5, random_state=RANDOM_STATE + i)
            en_b.fit(X_b, y_b)
            boot_preds[i] = en_b.predict(X_traintest)

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
            "min_samples_leaf": [2, 4, 6, 8, 10],
        }
        rf_cv = GridSearchCV(
            rf,
            cv_params,
            scoring="r2",
            # scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=4,
        )
        rf_cv.fit(X_train, y_train)

        best_rf = rf_cv.best_estimator_
        y_pred_rf = best_rf.predict(X_test)
        r2_rf = round(r2_score(y_test, y_pred_rf), 2)
        mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 1)

        y_pred_rf_full = best_rf.predict(X_traintest)
        # ---------- Jackknife-Varianz & PI -------------------------------------
        print("Computing jackknife variance for Random Forest...")
        rf_var = fci.random_forest_error(best_rf, X_train.shape, X_traintest)
        rf_sigma = np.sqrt(rf_var)
        rf_lower = y_pred_rf_full - 1.96 * rf_sigma
        rf_upper = y_pred_rf_full + 1.96 * rf_sigma

        evaluate_and_plot_parity(
            RESULTS_DIR, y_test, y_pred_rf, r2_rf, mae_rf, pseudonym, "rf", "03"
        )

        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X_test)
        plot_shap_summary_and_bar(
            RESULTS_DIR, shap_values, X_test, feature_names, pseudonym
        )

        mean_shap = np.abs(shap_values).mean(axis=0)
        for feature, shap_val in zip(feature_names, mean_shap):
            rf_feature_importances[feature].append(shap_val)
        top_indices = mean_shap.argsort()[::-1][:TOP_K]
        top_rf_features = [feature_names[i] for i in top_indices]
        rf_feature_counts.update(top_rf_features)

        raw_samples_not_covered_by_elastic_pi_ser = pd.Series(
            (y_traintest < en_lower) | (y_traintest > en_upper)
        )
        percent_raw_samples_not_covered_by_elastic_prediction_interval = (
            raw_samples_not_covered_by_elastic_pi_ser.value_counts().get(True, 0)
            / len(y_traintest)
            * 100
        )
        percent_raw_samples_not_covered_by_elastic_prediction_interval_train = (
            raw_samples_not_covered_by_elastic_pi_ser.loc[full_train_mask]
            .value_counts()
            .get(True, 0)
            / len(y_traintest)
            * 100
        )
        percent_raw_samples_not_covered_by_elastic_prediction_interval_test = (
            raw_samples_not_covered_by_elastic_pi_ser.loc[full_test_mask]
            .value_counts()
            .get(True, 0)
            / len(y_traintest)
            * 100
        )

        raw_samples_not_covered_by_rf_pi_ser = pd.Series(
            (y_traintest < rf_lower) | (y_traintest > rf_upper)
        )
        percent_raw_samples_not_covered_by_rf_prediction_interval = (
            raw_samples_not_covered_by_rf_pi_ser.value_counts().get(True, 0)
            / len(y_traintest)
            * 100
        )
        percent_raw_samples_not_covered_by_rf_prediction_interval_train = (
            raw_samples_not_covered_by_rf_pi_ser.loc[full_train_mask]
            .value_counts()
            .get(True, 0)
            / len(y_traintest)
            * 100
        )
        percent_raw_samples_not_covered_by_rf_prediction_interval_test = (
            raw_samples_not_covered_by_rf_pi_ser.loc[full_test_mask]
            .value_counts()
            .get(True, 0)
            / len(y_traintest)
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
            "elastic_rsquared_and_mae_criteria": None,
            "rf_rsquared_and_mae_criteria": None,
        }
        plot_data[pseudonym] = {
            "timestamps": timestamps_model,
            "phq2_raw": y_traintest.tolist(),
            "train_mask": full_train_mask.tolist(),
            "test_mask": full_test_mask.tolist(),
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
        results[pseudonym]["elastic_rsquared_and_mae_criteria"] = (
            results[pseudonym]["r2_elastic"] >= 0.3
            and results[pseudonym]["mae_elastic"] <= 2
        )
        results[pseudonym]["rf_rsquared_and_mae_criteria"] = (
            results[pseudonym]["r2_rf"] >= 0.3 and results[pseudonym]["mae_rf"] <= 2
        )

    # Filter importances based on criteria for elastic net
    valid_pseudonyms = {
        p
        for p, res in results.items()
        if res.get("elastic_rsquared_and_mae_criteria", True)
    }
    print(
        f"Valid pseudonyms after filtering for elastic net: {len(valid_pseudonyms)} of {len(results)}"
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

    # Filter importances based on criteria for random forest
    valid_pseudonyms = {
        p for p, res in results.items() if res.get("rf_rsquared_and_mae_criteria", True)
    }
    print(
        f"Valid pseudonyms after filtering for random forest: {len(valid_pseudonyms)} of {len(results)}"
    )

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
        RESULTS_DIR, elasticnet_feature_importances, rf_feature_importances
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
df_raw = pd.read_pickle("data/df_merged_v3.pickle")

# Map IDs to pseudonyms
json_file_path = "config/id_to_pseudonym.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    id_to_pseudonym = json.load(f)

df_raw["pseudonym"] = df_raw["patient_id"].map(id_to_pseudonym)
df_raw = df_raw.dropna(subset=["pseudonym"])
df_raw = df_raw.drop(columns=["patient_id"])

target_column = "abend_PHQ2_sum"
pseudonyms = df_raw["pseudonym"].unique()

TOP_K = 15
RANDOM_STATE = 42
CACHE_RESULTS_PATH = f"{RESULTS_DIR}/process_participants_results.pkl"

# --- Load or compute process_participants results ---
if os.path.exists(CACHE_RESULTS_PATH):
    print(f"[Info] Loading cached results from {CACHE_RESULTS_PATH} ...")
    cached_data = pd.read_pickle(CACHE_RESULTS_PATH)
    (
        results,
        elastic_counts,
        rf_counts,
        elastic_stats,
        rf_stats,
        plot_data,
    ) = cached_data
else:
    print("[Info] Computing results using process_participants ...")
    (
        results,
        elastic_counts,
        rf_counts,
        elastic_stats,
        rf_stats,
        plot_data,
    ) = process_participants(df_raw, pseudonyms, target_column)

    print(f"[Info] Saving results to {CACHE_RESULTS_PATH} ...")
    pd.to_pickle(
        (results, elastic_counts, rf_counts, elastic_stats, rf_stats, plot_data),
        CACHE_RESULTS_PATH,
    )


# %% Plot MAE and RMSSD for each participant
plot_mae_rmssd_bar_2(RESULTS_DIR, results, model_key="elastic")
plot_mae_rmssd_bar_2(RESULTS_DIR, results, model_key="rf")

# %% Plot PHQ-2 time series with predictions
plot_phq2_timeseries_from_results(RESULTS_DIR, plot_data, "elastic")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "elastic")
plot_phq2_test_errors_from_results(
    RESULTS_DIR,
    plot_data,
    "elastic",
    show_pred_ci=False,
)
plot_phq2_timeseries_from_results(RESULTS_DIR, plot_data, "rf")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf")
plot_phq2_test_errors_from_results(
    RESULTS_DIR,
    plot_data,
    "rf",
    show_pred_ci=False,
)

# %% Save evaluation metrics
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv(f"{RESULTS_DIR}/model_performance_summary.csv")

# Save feature importance counts
pd.Series(elastic_counts).sort_values(ascending=False).to_csv(
    f"{RESULTS_DIR}/elasticnet_top_features.csv"
)
pd.Series(rf_counts).sort_values(ascending=False).to_csv(
    f"{RESULTS_DIR}/randomforest_top_features.csv"
)

# Save feature importance mean and stds
pd.DataFrame(elastic_stats).T.sort_values("mean", ascending=False).to_csv(
    f"{RESULTS_DIR}/elasticnet_feature_importance_stats.csv"
)
pd.DataFrame(rf_stats).T.sort_values("mean", ascending=False).to_csv(
    f"{RESULTS_DIR}/randomforest_feature_importance_stats.csv"
)
