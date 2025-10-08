# %% Data Loading and Setup =====================================================
import json
import os
from collections import Counter
from datetime import timedelta
from typing import List, Tuple, Dict, Callable

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.dummy import DummyRegressor

set_config(transform_output="pandas")

# ==== RUN/PROJECT CONFIG ======================================================
CONFIG = {
    "random_state": 42,
    "top_k": 15,

    "paths": {
        "data": "data/df_merged_v7.pickle",
        "results_dir": "results_with_nightfeatures_perfeaturescaler_timeaware",
    },

    "targets": {
        "phq2": "abend_PHQ2_sum",
        "phq9": "woche_PHQ9_sum",
    },

    # 70:30 chronologischer Holdout
    "holdout_test_frac": 0.30,

    # GridSearchCV (TimeSeriesSplit)
    "cv_n_splits": 5,

    # Elastic Net 
    "en": {
        "scoring": "r2", # "neg_mean_absolute_error" # MAE vs. r2:  r2 "bestraft" mehr konstante verläufe.
        "l1_ratio": 0.5, 
        "max_iter": 50000,
        "tol": 1e-2, 
        "param_grid": {
            "model__alpha": list(np.logspace(-3, 2, 12)), # 0.000001 bis 10 weniger schrumpfen
        }
    },

    # Random Forest 
    "rf": {
        "scoring": "r2",
        "param_grid": {
            "model__max_depth": [4, 6, 8, 10, 15, 20, None],
            "model__n_estimators": [100, 200, 400, 800],
            "model__min_samples_leaf": [1, 2, 4, 6, 8],
        }
    },

    # Rolling Permutation Importance
    "pi": {
        "n_splits": 5,
        "test_size": 30,
        "min_train_size": 90,
        "embargo": 0,
        "n_repeats": 10, # 10 nunr für smoke test (sonst 50)
        "agg": "mean"  # "mean" oder "median"
    },

    # Rolling SHAP (gleiche Fenster wie PI)
    "shap": {
        "n_splits": 5,
        "test_size": 30,
        "min_train_size": 90,
        "embargo": 0,
        "agg": "mean"  # "mean" oder "median"
    },
}

# Ordner und Cachepfad
os.makedirs(CONFIG["paths"]["results_dir"], exist_ok=True)
CONFIG["paths"]["cache_results"] = os.path.join(
    CONFIG["paths"]["results_dir"], "process_participants_results.pkl"
)
with open(os.path.join(CONFIG["paths"]["results_dir"], "config.json"), "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

# DEFS
DATA_PATH = CONFIG["paths"]["data"]
RESULTS_DIR = CONFIG["paths"]["results_dir"]
CACHE_RESULTS_PATH = CONFIG["paths"]["cache_results"]
TOP_K = CONFIG["top_k"]
RANDOM_STATE = CONFIG["random_state"]
PHQ2_COLUMN = CONFIG["targets"]["phq2"]
PHQ9_COLUMN = CONFIG["targets"]["phq9"]

feature_config = pd.read_csv("config/feature_config.csv")
LOG1P_COLS = feature_config[feature_config["scaler"] == "log1p"]["feature"].tolist()
ZSCORE_COLS = feature_config[feature_config["scaler"] == "zscore"]["feature"].tolist()
MINMAX_COLS = feature_config[feature_config["scaler"] == "minmax"]["feature"].tolist()
BASELINE_FEATURES = feature_config[feature_config["baseline"] == 1]["feature"].tolist()
DAY_FEATURES = feature_config[feature_config["day"] == 1]["feature"].tolist()
NIGHT_FEATURES = feature_config[feature_config["night"] == 1]["feature"].tolist()
ACR_FEATURES = feature_config[feature_config["acr"] == 1]["feature"].tolist()

FEATURES_TO_CONSIDER = BASELINE_FEATURES + DAY_FEATURES + NIGHT_FEATURES + ACR_FEATURES

# ============ Feature-spezifische Scaler ======================================
class Log1pScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Namen für get_feature_names_out merken (falls DataFrame)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = None
        # Anzahl Features merken, falls später keine Namen verfügbar
        X_arr = np.asarray(X)
        self.n_features_in_ = X_arr.shape[1] if X_arr.ndim == 2 else 1
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        # NaN/Inf neutralisieren
        X_arr[~np.isfinite(X_arr)] = np.nan
        # negative auf 0 clippen (log1p robust)
        X_arr = np.clip(X_arr, 0, None)
        out = np.log1p(X_arr)
        return out  

    def inverse_transform(self, X):
        return np.expm1(X)

    def get_feature_names_out(self, input_features=None):
        # Sklearn ruft das beim ColumnTransformer zusammen
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            return self.feature_names_in_
        # Fallback: generische Namen
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


# %% Plot-Imports (deine vorhandenen Utils) ==========================
from plotting import (
    evaluate_and_plot_parity,
    plot_feature_importance_stats,
    plot_mae_rmssd_bar_2,
    plot_phq2_test_errors_from_results,
    plot_phq2_timeseries_from_results,
    plot_shap_summary_and_bar,
    plot_timeseries_with_folds,
)

# %% Utility and Pipeline Functions ============================================
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
    df = df.drop(columns=["pseudonym", "timestamp_utc"], errors="ignore").astype(float)
    y = df[target_column].values
    X = df.drop(columns=[target_column])
    X = replace_outliers_with_nan(X, multiplier=1.5)
    return X, y, X.columns

feature_scalers = {}
for col in list(set(LOG1P_COLS).intersection(set(FEATURES_TO_CONSIDER))):
    feature_scalers[col] = Log1pScaler()
for col in list(set(ZSCORE_COLS).intersection(set(FEATURES_TO_CONSIDER))):
    feature_scalers[col] = StandardScaler()
for col in list(set(MINMAX_COLS).intersection(set(FEATURES_TO_CONSIDER))):
    feature_scalers[col] = MinMaxScaler()

pre_scalers = ColumnTransformer(
    transformers=[(f"scale_{col}", scaler, [col]) for col, scaler in feature_scalers.items()],
    remainder=StandardScaler(),
    verbose_feature_names_out=False,
)

def preprocess_pipeline():
    # Imputer -> per-Feature-Scaler -> globale z-Skalierung -> finaler Imputer (0.0)
    return Pipeline([
        ("imputer", IterativeImputer(max_iter=50,tol=1e-2,n_nearest_features=30,random_state=RANDOM_STATE, keep_empty_features=True)),
        ("meanimputer", SimpleImputer(strategy="mean")), # only do somthing if IterativeImputer produces NaNs
        ("per_feature_scalers", pre_scalers),
        ("scaler", StandardScaler()),
        #("final_imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
    ])

def compute_rmssd(series):
    values = pd.Series(series).astype(float).to_numpy()
    diffs = np.diff(values)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    return np.sqrt(np.nanmean(diffs**2))

# %% ZEIT-SPLITS (Rolling-Origin, expanding) ============================
def rolling_origin_splits(
    n_samples: int,
    n_splits: int = 5,
    test_size: int = 30,
    embargo: int = 0,
    min_train_size: int = 90,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding Window:
      Fold 1: train[0 : T1] -> test[T1+embargo : T1+embargo+test_size]
      Fold 2: train[0 : T2] -> test[T2+embargo : T2+embargo+test_size]
      ...
    """
    splits = []
    max_last_test_end = n_samples - test_size
    if max_last_test_end <= min_train_size:
        return splits
    train_ends = np.linspace(min_train_size, max_last_test_end - embargo, n_splits, dtype=int)
    for tr_end in train_ends:
        tr_idx = np.arange(0, tr_end)
        start_test = tr_end + embargo
        end_test = min(start_test + test_size, n_samples)
        if end_test - start_test <= 0:
            continue
        te_idx = np.arange(start_test, end_test)
        if len(tr_idx) >= min_train_size and len(te_idx) >= 5:
            splits.append((tr_idx, te_idx))
    return splits

# %%  PERMUTATION-IMPORTANCE (modell-agnostisch) =========================
def permutation_importance_per_feature(
    fitted_estimator: BaseEstimator,
    X_test: pd.DataFrame,   # PREPROCESSED DataFrame (Feature-Namen wichtig)
    y_test: np.ndarray,
    n_repeats: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    y_hat_base = fitted_estimator.predict(X_test)
    R2_base = r2_score(y_test, y_hat_base)
    MAE_base = mean_absolute_error(y_test, y_hat_base)

    rows = []
    cols = list(X_test.columns)
    for j, col in enumerate(cols):
        dR2_list, dMAE_list = [], []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            Xp.iloc[:, j] = Xp.iloc[perm, j].to_numpy()
            y_hat = fitted_estimator.predict(Xp)
            R2_p = r2_score(y_test, y_hat)
            MAE_p = mean_absolute_error(y_test, y_hat)
            dR2_list.append(R2_base - R2_p)
            dMAE_list.append(MAE_p - MAE_base)
        rows.append({
            "feature": str(col).split("__")[-1],
            "delta_R2": float(np.mean(dR2_list)),
            "delta_MAE": float(np.mean(dMAE_list)),
        })
    return pd.DataFrame(rows).sort_values("delta_R2", ascending=False).reset_index(drop=True)

def run_feature_permutation_importance_over_splits(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    test_size: int,
    embargo: int,
    min_train_size: int,
    estimator: BaseEstimator,    # bereits mit finalen HP konfiguriert
    n_repeats: int = 50,
    agg: str = "mean",
    random_state: int = 42,
):
    n = len(y)
    splits = rolling_origin_splits(
        n_samples=n, n_splits=n_splits, test_size=test_size,
        embargo=embargo, min_train_size=min_train_size,
    )

    all_rows = []
    for fold_id, (tr, te) in enumerate(splits, start=1):
        pp = preprocess_pipeline()
        Xt_tr = pp.fit_transform(X_df.iloc[tr])
        Xt_te = pp.transform(X_df.iloc[te])

        est = clone(estimator)
        est.fit(Xt_tr, y[tr])

        df_imp = permutation_importance_per_feature(
            fitted_estimator=est,
            X_test=Xt_te,
            y_test=y[te],
            n_repeats=n_repeats,
            random_state=random_state,
        )
        df_imp["fold"] = fold_id
        all_rows.append(df_imp)

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    imp_folds = pd.concat(all_rows, ignore_index=True)

    if agg == "median":
        agg_fun = "median"
    else:
        agg_fun = "mean"

    imp_summary = (
        imp_folds.groupby("feature")[["delta_R2", "delta_MAE"]]
        .agg(agg_fun)
        .sort_values("delta_R2", ascending=False)
        .reset_index()
    )
    return imp_folds, imp_summary

# %% ROLLING SHAP (modell-agnostisch) ===================================
def pick_shap_explainer(estimator: BaseEstimator, kind: str, Xt_train: pd.DataFrame):
    """Wählt den passenden SHAP-Explainer."""
    if kind == "linear":
        return shap.LinearExplainer(estimator, Xt_train)
    if kind == "tree":
        return shap.TreeExplainer(estimator)
    # auto
    name = estimator.__class__.__name__.lower()
    if "randomforest" in name or "gradientboost" in name or "xgb" in name or "lgbm" in name:
        return shap.TreeExplainer(estimator)
    if "elasticnet" in name or "lasso" in name or "ridge" in name or "linear" in name:
        return shap.LinearExplainer(estimator, Xt_train)
    return shap.Explainer(estimator, Xt_train)

def run_shap_over_splits(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    test_size: int,
    embargo: int,
    min_train_size: int,
    estimator: BaseEstimator,      # mit finalen HP konfiguriert
    explainer_kind: str = "auto",  # "auto" | "linear" | "tree"
    agg: str = "mean",
    random_state: int = 42,
):
    splits = rolling_origin_splits(
        n_samples=len(y),
        n_splits=n_splits,
        test_size=test_size,
        embargo=embargo,
        min_train_size=min_train_size,
    )

    rows = []
    for fold_id, (tr, te) in enumerate(splits, start=1):
        pp = preprocess_pipeline()
        Xt_tr = pp.fit_transform(X_df.iloc[tr])
        Xt_te = pp.transform(X_df.iloc[te])

        est = clone(estimator)
        est.fit(Xt_tr, y[tr])

        try:
            explainer = pick_shap_explainer(est, explainer_kind, Xt_tr)
            shap_vals = explainer.shap_values(Xt_te)
        except Exception:
            expl = shap.Explainer(est, Xt_tr)
            out = expl(Xt_te)
            shap_vals = getattr(out, "values", out)

        cols = [str(c).split("__")[-1] for c in list(Xt_te.columns)]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)
        for f, a, s in zip(cols, mean_abs, mean_signed):
            rows.append({"fold": fold_id, "feature": f,
                         "mean_abs_shap": float(a),
                         "mean_signed_shap": float(s)})

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    shap_folds = pd.DataFrame(rows)
    if agg == "median":
        agg_fun = "median"
    else:
        agg_fun = "mean"

    shap_summary = (
        shap_folds.groupby("feature")[["mean_abs_shap", "mean_signed_shap"]]
        .agg(agg_fun)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index()
    )
    return shap_folds, shap_summary

def evaluate_over_splits(
    X_df, y, make_model,                 
    n_splits=5, test_size=30, embargo=0, min_train_size=90,
    timestamps=None, model_label="model"
):
    """
    Pro Fold: Preprocessing NUR auf Train fitten, Modell trainieren, auf Test bewerten.
    Gibt DataFrame mit (fold, train_size, test_size, fold_end_idx/ts, R2, MAE,
    R2_baseline_last, R2_baseline_mean) zurück.
    """
    splits = rolling_origin_splits(
        n_samples=len(y), n_splits=n_splits, test_size=test_size,
        embargo=embargo, min_train_size=min_train_size
    )

    rows = []
    for fold_id, (tr, te) in enumerate(splits, 1):
        pp = preprocess_pipeline()
        Xt_tr = pp.fit_transform(X_df.iloc[tr])
        Xt_te = pp.transform(X_df.iloc[te])

        model = make_model()
        model.fit(Xt_tr, y[tr])
        y_hat = model.predict(Xt_te)

        # Naive Baselines (konstant)
        y_base_last = np.full_like(y[te], fill_value=y[tr][-1], dtype=float)
        y_base_mean = np.full_like(y[te], fill_value=np.mean(y[tr]), dtype=float)

        rows.append({
            "model": model_label,
            "fold": fold_id,
            "train_size": len(tr),
            "test_size": len(te),
            "fold_start_idx": int(te[0]),
            "fold_end_idx": int(te[-1]),
            "fold_end_ts": (str(timestamps[te[-1]]) if timestamps is not None else None),
            "r2": float(r2_score(y[te], y_hat)),
            "mae": float(mean_absolute_error(y[te], y_hat)),
            "r2_baseline_last": float(r2_score(y[te], y_base_last)),
            "r2_baseline_mean": float(r2_score(y[te], y_base_mean)),
        })

    return pd.DataFrame(rows)


# %% Main Processing Function ===================================================
def days_since_start(group):
    start_date = group["timestamp_utc"].min()
    return (group["timestamp_utc"] - start_date).dt.days

def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    plot_data = {}

    df_raw["day_of_week"] = df_raw["timestamp_utc"].dt.weekday
    df_raw["month_of_year"] = df_raw["timestamp_utc"].dt.month
    df_raw["day_since_start"] = df_raw.groupby("pseudonym", group_keys=False).apply(days_since_start)

    ensure_results_dir()

    for pseudonym in pseudonyms:
        print(f"Processing {pseudonym}")
        df_participant = df_raw[df_raw["pseudonym"] == pseudonym].iloc[:365].copy()
        df_participant_timeaware = df_participant.set_index("timestamp_utc").copy()
        y_rolling_mean = df_participant_timeaware[target_column].rolling(window=timedelta(days=14)).mean()

        rmssd_phq2 = compute_rmssd(df_participant[PHQ2_COLUMN].values)
        rmssd_values_phq2.append(rmssd_phq2)

        # --- Daten vorbereiten (nur gültige Zielwerte) ---
        X, y, feature_names = prepare_data(df_participant, target_column)
        mask_valid = np.isfinite(y)
        X = X.loc[mask_valid].reset_index(drop=True)
        y = np.asarray(y)[mask_valid]
        n = len(y)
        if n < 30:
            print(f"[WARN] {pseudonym}: zu wenige Datenpunkte ({n}), überspringe.")
            continue

        # --- 70:30 HOLDOUT (chronologisch, kein Shuffle) ---
        test_frac = CONFIG["holdout_test_frac"]
        gap = 0
        split = int(round((1 - test_frac) * n))
        idx_train = np.arange(0, split - gap)
        idx_test = np.arange(split, n)

        X_train_df = X.iloc[idx_train].copy()
        X_test_df = X.iloc[idx_test].copy()
        y_train = y[idx_train]
        y_test = y[idx_test]

        print("var(y_train) =", float(np.var(y_train)))
        print("var(y_test) =", float(np.var(y_test)))
        dum = DummyRegressor(strategy="mean").fit(X_train_df, y_train)
        print("baseline R2 =", r2_score(y_test, dum.predict(X_test_df)))

        # =================== ELASTIC NET: GridSearch + SHAP ====================
        print("  [EN] GridSearchCV (TimeSeriesSplit) ...")

        ## Alternative 1: GridSearch mit ElasticNet (ohne CV im Modell)
        # en_pipe = Pipeline([
        #     ("pre", preprocess_pipeline()),
        #     ("model", ElasticNet(max_iter=CONFIG["en"]["max_iter"],tol=CONFIG["en"]["tol"], l1_ratio=CONFIG["en"]["l1_ratio"], random_state=RANDOM_STATE)),
        # ])
        # en_gs = GridSearchCV(
        #     estimator=en_pipe,
        #     param_grid=CONFIG["en"]["param_grid"],
        #     scoring=CONFIG["en"]["scoring"],
        #     cv=TimeSeriesSplit(n_splits=CONFIG["cv_n_splits"]),
        #     n_jobs=-1,
        #     refit=True,
        # )
        # en_gs.fit(X_train_df, y_train)
        # best_en: Pipeline = en_gs.best_estimator_

        en_pipe = Pipeline([
            ("pre", preprocess_pipeline()),
            ("model", ElasticNetCV(
                l1_ratio=0.5,
                alphas=np.logspace(-3, 2, 50),
                cv=TimeSeriesSplit(n_splits=CONFIG["cv_n_splits"]),
                max_iter=200_000,
                tol=1e-2,
                random_state=RANDOM_STATE,
            )),
        ])
        en_pipe.fit(X_train_df, y_train)
        best_en = en_pipe  # enthält bereits das beste Alpha im Schritt "model"


        # Holdout-Performance (Prospektiv)
        y_pred_en = best_en.predict(X_test_df)
        r2_elastic = round(r2_score(y_test, y_pred_en), 2)
        mae_elastic = round(mean_absolute_error(y_test, y_pred_en), 1)

        # ElasticNet-Modell extrahieren
        model_en = best_en.named_steps["model"]
        best_alpha = float(model_en.alpha_)
        best_l1 = float(model_en.l1_ratio_)

        # SHAP auf preprocessed Matrizen
        pre_en = best_en.named_steps["pre"]
        Xt_train_en = pre_en.transform(X_train_df)
        Xt_test_en = pre_en.transform(X_test_df)
        
        try:
            en_explainer = shap.LinearExplainer(model_en, Xt_train_en)
            en_shap_values = en_explainer.shap_values(Xt_test_en)
        except Exception:
            expl = shap.Explainer(model_en, Xt_train_en)
            out = expl(Xt_test_en)
            en_shap_values = getattr(out, "values", out)

        print(f"EN R2={r2_elastic}, MAE={mae_elastic}, alpha={best_alpha}, l1_ratio={best_l1}")
        print("EN # nonzero betas:", np.count_nonzero(model_en.coef_))
        print("mean(|SHAP|):", float(np.nanmean(np.abs(en_shap_values))))

        en_feat_names = [str(c).split("__")[-1] for c in list(Xt_test_en.columns)]
        plot_shap_summary_and_bar(RESULTS_DIR, en_shap_values, Xt_test_en, en_feat_names, f"{pseudonym}_EN")

        # Modelle speichern
        joblib.dump(best_en, os.path.join(RESULTS_DIR, "models", f"{pseudonym}_elasticnet.joblib"))

        # ================= RANDOM FOREST: GridSearch + SHAP ====================
        print("  [RF] GridSearchCV (TimeSeriesSplit) ...")
        rf_pipe = Pipeline([
            ("pre", preprocess_pipeline()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
        ])
        rf_gs = GridSearchCV(
            estimator=rf_pipe,
            param_grid=CONFIG["rf"]["param_grid"],
            scoring=CONFIG["rf"]["scoring"],
            cv=TimeSeriesSplit(n_splits=CONFIG["cv_n_splits"]),
            n_jobs=-1,
            refit=True,
        )
        rf_gs.fit(X_train_df, y_train)
        best_rf: Pipeline = rf_gs.best_estimator_

        y_pred_rf = best_rf.predict(X_test_df)
        r2_rf = round(r2_score(y_test, y_pred_rf), 2)
        mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 1)

        pre_rf = best_rf.named_steps["pre"]
        Xt_test_rf = pre_rf.transform(X_test_df)
        model_rf = best_rf.named_steps["model"]
        rf_explainer = shap.TreeExplainer(model_rf)
        rf_shap_values = rf_explainer.shap_values(Xt_test_rf)
        rf_feat_names = [str(c).split("__")[-1] for c in list(Xt_test_rf.columns)]
        plot_shap_summary_and_bar(RESULTS_DIR, rf_shap_values, Xt_test_rf, rf_feat_names, f"{pseudonym}_RF")

        print(f"RF R2={r2_rf}, MAE={mae_rf}, n_estimators={model_rf.n_estimators}, max_depth={model_rf.max_depth}, min_samples_leaf={model_rf.min_samples_leaf}")
        print("mean(|SHAP|):", float(np.nanmean(np.abs(rf_shap_values))))
        print("Wichtigste Features (SHAP):", rf_feat_names[np.argsort(-np.abs(rf_shap_values).mean(0))[:5]])
        # Modelle speichern
        joblib.dump(best_rf, os.path.join(RESULTS_DIR, "models", f"{pseudonym}_rf.joblib"))

        # ==================== Parity Plots & Zeitreihen-Plots ==================
        evaluate_and_plot_parity(RESULTS_DIR, y_test, y_pred_en, r2_elastic, mae_elastic, pseudonym, "elasticnet", "01")
        evaluate_and_plot_parity(RESULTS_DIR, y_test, y_pred_rf, r2_rf, mae_rf, pseudonym, "rf", "03")

        # Für Zeitreihen-Plot Roh/Pred (optional: nur Holdout anzeigen)
        timestamps_model = df_participant.loc[mask_valid, "timestamp_utc"].astype(str).tolist()
        plot_data[pseudonym] = {
            "timestamps": timestamps_model,
            "phq2_raw": y.tolist(),
            "train_mask": [i < split for i in range(n)],
            "test_mask": [i >= split for i in range(n)],
            "elastic": {"pred": list(best_en.predict(X)), "lower": None, "upper": None},
            "rf": {"pred": list(best_rf.predict(X)), "lower": None, "upper": None},
        }

        # ==================== Rolling PI & Rolling SHAP ========================
        print("  [EN] Rolling PI/SHAP ...")
        # final HP für EN aus GridSearch
        en_final = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000, random_state=RANDOM_STATE)

        imp_folds_en, imp_sum_en = run_feature_permutation_importance_over_splits(
            X_df=X, y=y,
            n_splits=CONFIG["pi"]["n_splits"],
            test_size=CONFIG["pi"]["test_size"],
            embargo=CONFIG["pi"]["embargo"],
            min_train_size=CONFIG["pi"]["min_train_size"],
            estimator=en_final,
            n_repeats=CONFIG["pi"]["n_repeats"],
            agg=CONFIG["pi"]["agg"],
            random_state=RANDOM_STATE,
        )
        shap_folds_en, shap_sum_en = run_shap_over_splits(
            X_df=X, y=y,
            n_splits=CONFIG["shap"]["n_splits"],
            test_size=CONFIG["shap"]["test_size"],
            embargo=CONFIG["shap"]["embargo"],
            min_train_size=CONFIG["shap"]["min_train_size"],
            estimator=en_final,
            explainer_kind="linear",
            agg=CONFIG["shap"]["agg"],
            random_state=RANDOM_STATE,
        )
        imp_folds_en.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_EN_perm_feat_folds.csv"), index=False)
        imp_sum_en.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_EN_perm_feat_summary.csv"), index=False)
        shap_folds_en.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_EN_shap_folds.csv"), index=False)
        shap_sum_en.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_EN_shap_summary.csv"), index=False)

        print("  [RF] Rolling PI/SHAP ...")
        # final HP für RF aus GridSearch
        rf_params = {
            "n_estimators": best_rf.get_params()["model__n_estimators"],
            "max_depth": best_rf.get_params()["model__max_depth"],
            "min_samples_leaf": best_rf.get_params()["model__min_samples_leaf"],
            "random_state": RANDOM_STATE,
        }
        rf_final = RandomForestRegressor(**rf_params)

        imp_folds_rf, imp_sum_rf = run_feature_permutation_importance_over_splits(
            X_df=X, y=y,
            n_splits=CONFIG["pi"]["n_splits"],
            test_size=CONFIG["pi"]["test_size"],
            embargo=CONFIG["pi"]["embargo"],
            min_train_size=CONFIG["pi"]["min_train_size"],
            estimator=rf_final,
            n_repeats=CONFIG["pi"]["n_repeats"],
            agg=CONFIG["pi"]["agg"],
            random_state=RANDOM_STATE,
        )
        shap_folds_rf, shap_sum_rf = run_shap_over_splits(
            X_df=X, y=y,
            n_splits=CONFIG["shap"]["n_splits"],
            test_size=CONFIG["shap"]["test_size"],
            embargo=CONFIG["shap"]["embargo"],
            min_train_size=CONFIG["shap"]["min_train_size"],
            estimator=rf_final,
            explainer_kind="tree",
            agg=CONFIG["shap"]["agg"],
            random_state=RANDOM_STATE,
        )
        imp_folds_rf.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_RF_perm_feat_folds.csv"), index=False)
        imp_sum_rf.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_RF_perm_feat_summary.csv"), index=False)
        shap_folds_rf.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_RF_shap_folds.csv"), index=False)
        shap_sum_rf.to_csv(os.path.join(RESULTS_DIR, f"{pseudonym}_RF_shap_summary.csv"), index=False)

        # ==================== Ergebnisse sammeln ===============================
        results[pseudonym] = {
            "r2_elastic": r2_elastic,
            "mae_elastic": mae_elastic,
            "r2_rf": r2_rf,
            "mae_rf": mae_rf,
            "rmssd_phq2": rmssd_phq2,
            "en_best_alpha": best_alpha,
            "en_best_l1_ratio": best_l1,
            "rf_best_n_estimators": rf_params["n_estimators"],
            "rf_best_max_depth": rf_params["max_depth"],
            "rf_best_min_samples_leaf": rf_params["min_samples_leaf"],
        }
        
    # (optional) globale Schwellen/Flags etc. weglassen – Fokus auf Metriken & Importances
    return results, plot_data

# %% Run the pipeline ===========================================================
# Load dataset
df_raw = pd.read_pickle(DATA_PATH)
df_raw["day_of_week"] = df_raw["timestamp_utc"].dt.weekday
df_raw["month_of_year"] = df_raw["timestamp_utc"].dt.month
df_raw["day_since_start"] = df_raw.groupby("patient_id", group_keys=False).apply(days_since_start)

df_raw = df_raw[["patient_id", "timestamp_utc", PHQ2_COLUMN] + FEATURES_TO_CONSIDER]

# Map IDs to pseudonyms
json_file_path = "config/id_to_pseudonym.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    id_to_pseudonym = json.load(f)

df_raw["pseudonym"] = df_raw["patient_id"].map(id_to_pseudonym)
df_raw = df_raw.dropna(subset=["pseudonym"])
df_raw = df_raw.drop(columns=["patient_id"])

target_column = PHQ2_COLUMN
pseudonyms = df_raw["pseudonym"].unique()

if os.path.exists(CACHE_RESULTS_PATH):
    print(f"[Info] Loading cached results from {CACHE_RESULTS_PATH} ...")
    cached_data = pd.read_pickle(CACHE_RESULTS_PATH)
    results, plot_data = cached_data
else:
    print("[Info] Computing results using process_participants ...")
    results, plot_data = process_participants(df_raw, pseudonyms, target_column)
    print(f"[Info] Saving results to {CACHE_RESULTS_PATH} ...")
    pd.to_pickle((results, plot_data), CACHE_RESULTS_PATH)

# %% Plots (MAE/RMSSD & Zeitreihen) ============================================
plot_mae_rmssd_bar_2(RESULTS_DIR, results, model_key="elastic")
plot_mae_rmssd_bar_2(RESULTS_DIR, results, model_key="rf")

plot_phq2_timeseries_from_results(RESULTS_DIR, plot_data, "elastic")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "elastic")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "elastic", show_pred_ci=False)

plot_phq2_timeseries_from_results(RESULTS_DIR, plot_data, "rf")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf", show_pred_ci=False)

# %% Save evaluation metrics (per participant) ==================================
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv(f"{RESULTS_DIR}/model_performance_summary.csv")
