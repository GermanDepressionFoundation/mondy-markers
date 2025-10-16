# %% Data Loading and Setup
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import shap
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

set_config(transform_output="pandas")


class Log1pScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # add parameters here if needed (e.g., shift)

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        return np.log1p(X)

    def inverse_transform(self, X):
        return np.expm1(X)


from plotting import (
    evaluate_and_plot_parity,
    plot_elasticnet_coefficients,
    plot_feature_importance_stats,
    plot_shap_summary_and_bar,
)

RESULTS_DIR = "results"
TOP_K = 15
RANDOM_STATE = 42
CACHE_RESULTS_PATH = f"{RESULTS_DIR}/process_participants_results.pkl"
PHQ2_COLUMN = "abend_PHQ2_sum"

# %% Utility and Pipeline Functions


def ensure_results_dir():
    os.makedirs(f"{RESULTS_DIR}/models", exist_ok=True)


def replace_outliers_with_nan(df, multiplier=1.5):
    df_out = df.copy()
    Q1 = df.quantile(0.01)
    Q3 = df.quantile(0.99)
    IQR = Q3 - Q1

    for col in df.columns:
        lower = Q1[col] - multiplier * IQR[col]
        upper = Q3[col] + multiplier * IQR[col]
        mask = (df[col] < lower) | (df[col] > upper)
        df_out.loc[mask, col] = np.nan
    return df_out


def prepare_data(df, target_column):
    df = df.drop(
        columns=["pseudonym", "timestamp_utc"],
        errors="ignore",
    )
    df = df.astype(float)
    y = df[target_column].values
    X = df.drop(columns=[target_column])
    X = replace_outliers_with_nan(X, multiplier=1.5)
    return X, y, X.columns


feature_config = pd.read_csv("config/feature_config.csv")
LOG1P_COLS = feature_config[feature_config["scaler"] == "log1p"]["feature"].tolist()
ZSCORE_COLS = feature_config[feature_config["scaler"] == "zscore"]["feature"].tolist()
MINMAX_COLS = feature_config[feature_config["scaler"] == "minmax"]["feature"].tolist()
BASELINE_FEATURES = feature_config[feature_config["baseline"] == 1]["feature"].tolist()
DAY_FEATURES = feature_config[feature_config["day"] == 1]["feature"].tolist()
NIGHT_FEATURES = feature_config[feature_config["night"] == 1]["feature"].tolist()
ACR_FEATURES = feature_config[feature_config["acr"] == 1]["feature"].tolist()

FEATURES_TO_CONSIDER = BASELINE_FEATURES + DAY_FEATURES + NIGHT_FEATURES + ACR_FEATURES

feature_scalers = {}
for col in list(set(LOG1P_COLS).intersection(set(FEATURES_TO_CONSIDER))):
    feature_scalers[col] = Log1pScaler()
for col in list(set(ZSCORE_COLS).intersection(set(FEATURES_TO_CONSIDER))):
    feature_scalers[col] = StandardScaler()
for col in list(set(MINMAX_COLS).intersection(set(FEATURES_TO_CONSIDER))):
    feature_scalers[col] = MinMaxScaler()

pre_scalers = ColumnTransformer(
    transformers=[
        (f"scale_{col}", scaler, [col]) for col, scaler in feature_scalers.items()
    ],
    remainder="passthrough",
)


def preprocess_pipeline():
    return Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    max_iter=20, random_state=RANDOM_STATE, keep_empty_features=True
                ),
            ),
            ("per_feature_scalers", pre_scalers),
        ]
    )


def days_since_start(group):
    start_date = group["timestamp_utc"].min()
    return (group["timestamp_utc"] - start_date).dt.days


def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    plot_data = {}

    elasticnet_feature_importances = {
        feature: []
        for feature in df_raw.columns
        if feature
        not in ["pseudonym", "timestamp_utc", "woche_PHQ9_sum", target_column]
    }
    rf_feature_importances = {
        feature: []
        for feature in df_raw.columns
        if feature
        not in ["pseudonym", "timestamp_utc", "woche_PHQ9_sum", target_column]
    }
    elasticnet_feature_counts = Counter()
    rf_feature_counts = Counter()

    ensure_results_dir()

    # --- Per-participant loop ---
    for pseudonym in pseudonyms:
        print(f"Processing {pseudonym}")
        df_participant = df_raw[df_raw["pseudonym"] == pseudonym].iloc[:365]

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
        X_train_transformed = pipeline.fit_transform(
            pd.DataFrame(X_train, columns=feature_names)
        )
        X_test_transformed = pipeline.transform(
            pd.DataFrame(X_test, columns=feature_names)
        )

        df = df_participant.reset_index(drop=True)
        train_mask = df.loc[idx_train, target_column].notna()
        test_mask = df.loc[idx_test, target_column].notna()

        X_train = X_train_transformed[train_mask.values]
        y_train = y_train[train_mask.values]
        X_test = X_test_transformed[test_mask.values]
        y_test = y_test[test_mask.values]

        cv = 5

        try:
            # --- Elastic Net ---
            print("Training Elastic Net...")
            elastic = ElasticNetCV(
                cv=cv,
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                alphas=None,  # default grid,
                random_state=RANDOM_STATE,
            )

            elastic.fit(
                X_train,
                y_train,
                # sample_weight=weights
            )
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

        # --- Random Forest ---
        print("Training Random Forest...")
        rf = RandomForestRegressor(random_state=RANDOM_STATE)
        cv_params = {
            "max_depth": [4, 6, 8, 10, 15, 20],
            "n_estimators": [75, 100, 125, 150, 175, 200, 500, 1000],
            "min_samples_leaf": [2, 4, 6, 8, 10],
            # "max_features": ["sqrt", 1],
        }
        rf_cv = GridSearchCV(
            rf,
            cv_params,
            scoring="r2",
            # scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=4,
        )
        rf_cv.fit(
            X_train,
            y_train,
            # sample_weight=weights
        )

        best_rf = rf_cv.best_estimator_
        y_pred_rf = best_rf.predict(X_test)
        r2_rf = round(r2_score(y_test, y_pred_rf), 2)
        mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 1)

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

        # Store per-participant results
        results[pseudonym] = {
            "r2_elastic": r2_elastic,
            "mae_elastic": mae_elastic,
            "r2_rf": r2_rf,
            "mae_rf": mae_rf,
            "low_variance_candidate": None,
            "elastic_mae_lower_rmssd_phq2": None,
            "rf_mae_lower_rmssd_phq2": None,
            "elastic_mae+10p_lower_rmssd_phq2": None,
            "rf_mae+10p_lower_rmssd_phq2": None,
            "elastic_rsquared_and_mae_criteria": None,
            "rf_rsquared_and_mae_criteria": None,
        }

    # Compute 25th percentile thresholds across all participants
    phq2_thresh = np.nanpercentile(rmssd_values_phq2, 25)
    print(f"25th percentile thresholds: PHQ-2={phq2_thresh:.3f}")

    # Flag participants as low-variance candidates
    for pseudonym in results:
        is_low_var = results[pseudonym]["rmssd_phq2"] < phq2_thresh
        results[pseudonym]["low_variance_candidate"] = is_low_var

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
            for i, (pseudonym, _) in enumerate(results.items())
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
            for i, (pseudonym, _) in enumerate(results.items())
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
target_column = PHQ2_COLUMN

df_raw = pd.read_pickle("data/df_merged_v7.pickle")
df_raw["day_of_week"] = df_raw["timestamp_utc"].dt.weekday
df_raw["month_of_year"] = df_raw["timestamp_utc"].dt.month
df_raw["day_since_start"] = df_raw.groupby("patient_id", group_keys=False).apply(
    days_since_start
)
df_raw = df_raw[["patient_id", "timestamp_utc", target_column] + FEATURES_TO_CONSIDER]

# Map IDs to pseudonyms
json_file_path = "config/id_to_pseudonym.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    id_to_pseudonym = json.load(f)

df_raw["pseudonym"] = df_raw["patient_id"].map(id_to_pseudonym)
df_raw = df_raw.dropna(subset=["pseudonym"])
df_raw = df_raw.drop(columns=["patient_id"])

pseudonyms = df_raw["pseudonym"].unique()


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
