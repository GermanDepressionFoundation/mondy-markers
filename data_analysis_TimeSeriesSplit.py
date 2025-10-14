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
        "results_dir": "results_with_nightfeatures_perfeaturescaler_timeaware_test",
    },

    "targets": {
        "phq2": "abend_PHQ2_sum",
        "phq9": "woche_PHQ9_sum",
    },

    # 70:30 chronologischer Holdout
    "holdout_test_frac": 0.70,

    # Anteil Startfenster für HP-Tuning und Fix-Scaler-Fit
    "init_frac_for_hp": 0.30, 

    # GridSearchCV (TimeSeriesSplit)
    "cv_n_splits": 5,

    # globalen Scaler einmal im Startfenster fitten?
    "use_fixed_scaler": True,

    # Elastic Net 
    "en": {
        "scoring": "r2", # "neg_mean_absolute_error" # MAE vs. r2:  r2 "bestraft" mehr konstante verläufe.
        "tune_l1_ratio": False,  # True: l1_ratio in [0.1, 0.5, 0.9] mit-tunen
        "l1_ratio": 0.5, 
        "max_iter": 50000,
        "tol": 1e-2, 
        "param_grid": {
            "model__alpha": list(np.logspace(-3, 2, 12)), # 0.000001 bis 10 weniger schrumpfen
        }, 
        "en_l1_grid": [0.0, 0.1, 0.3, 0.5],
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
        "agg": "mean",  # "mean" oder "median"
        "block_len": 7  # Länge der Blöcke für blockweise Permutation
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


# Groups für gruppenweise Permutation Importance (auf Quellenebene über "category")
def _feats(cats):
    return feature_config[feature_config["category"].isin(cats)]["feature"].tolist()

# Gruppen (ohne Day/Night)
COARSE_GROUPS = {
    "HRV":  _feats(["HRV"]),
    "ACR":  _feats(["steps", "ACR"]),          # Aktivität/Transitions
    "COMM": _feats(["app usage", "calls"]),    # Kommunikation / App-Use
    "AUDIO":_feats(["audio"]),                 # Sprach-/Stimm-Proxys
    "SLEEP":_feats(["sleep"]),
    "CAL":  _feats(["time"]),                  # Kalender/Trend
}

# Day/Night-Variante für Quellen
DAY_NIGHT_GROUPS = {}
for name, cats in {
    "HRV":  ["HRV"],
    "ACR":  ["steps", "ACR"],
    "COMM": ["app usage", "calls"],
    "SLEEP":["sleep"],
}.items():
    for period in ["day", "night"]:
        mask = feature_config["category"].isin(cats) & (feature_config[period] == 1)
        DAY_NIGHT_GROUPS[f"{name}_{period}"] = feature_config.loc[mask, "feature"].tolist()

#leere Gruppen entfernen
DAY_NIGHT_GROUPS = {k: v for k, v in DAY_NIGHT_GROUPS.items() if v}

CONFIG["feature_groups"] = COARSE_GROUPS

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
    plot_importance_bars,
    plot_folds_on_timeseries,
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

def preprocess_train_only():
    """
    Preprocessing-Schritte, die pro Fold auf dem Trainingsfenster gefittet werden.

    Enthält alle Schritte, die drift-/missingness-sensitiv sind (z. B. MICE-Imputation,
    per-Feature-Transformationen je Quelle). Diese werden in jedem Fold auf dem
    Trainingsblock gefittet und dann auf Train/Test angewandt – leakage-sicher.

    Returns
    -------
    sklearn.Pipeline
        Pipeline ohne den globalen, fixen StandardScaler.

    Notes
    -----
    - Der globale, fixe Scaler (falls genutzt) wird separat bereitgestellt.
    - So bleiben β/SHAP mit fixem Scaler über Zeit vergleichbar, während
      Imputation & per-Feature-Skalierung weiterhin train-only bleiben.
    """
    # Beispiel: baue hier deine bisherigen Steps OHNE finalen globalen StandardScaler
    # (Passe an deine realen Step-Namen/Objekte an!)
    return Pipeline([
        ("imputer_mice", IterativeImputer(max_iter=50, tol=1e-2, random_state=CONFIG["random_state"])),
        ("simple_imputer", SimpleImputer(strategy="mean")),
        ("per_feature_scalers", pre_scalers),  # dein bestehender ColumnTransformer je Quelle
        # KEIN globaler StandardScaler hier!
    ])

def fit_fixed_global_scaler(X_init: pd.DataFrame):
    """
    Fit eines *globalen* StandardScalers auf dem Startfenster.

    Dieser Scaler wird danach NICHT mehr refittet, sondern in allen Folds
    nur noch auf Train/Test angewandt. Damit werden Feature-Skalen über Zeit
    konsistent und β/SHAP vergleichbar.

    Parameters
    ----------
    X_init : pandas.DataFrame
        Featurematrix des Startfensters (nach train-only Preprocessing).

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Gefitteter StandardScaler (global, fix).

    Notes
    -----
    - Leakage-sicher, da nur auf *frühen* Daten gefittet.
    - In späteren Folds wird *zuerst* train-only Preprocessing angewandt,
      *dann* dieser feste Scaler transformiert.
    """
    scaler = StandardScaler()
    scaler.fit(X_init)
    return scaler

def split_initial_window(n_samples: int, frac: float, min_train_size: int) -> int:
    """
    Bestimmt das Ende des Startfensters für HP-Tuning & Fix-Scaler-Fit.

    Nimmt den größeren der beiden Werte: floor(frac * n_samples) und min_train_size,
    damit das Startfenster groß genug für stabiles Tuning/Scaling ist.

    Parameters
    ----------
    n_samples : int
        Anzahl der Zeilen (Tage).
    frac : float
        Anteil (z. B. 0.30 für 30 %).
    min_train_size : int
        Untergrenze in Tagen.

    Returns
    -------
    int
        Exklusiver Endindex (Python-Slicing) des Startfensters.

    Notes
    -----
    - Typischerweise 30 % oder min_train_size, was immer größer ist.
    """
    import math
    return max(int(math.floor(frac * n_samples)), int(min_train_size))

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

def permutation_importance_by_group(
    fitted_estimator, X_test, y_test, groups: Dict[str, list],
    n_repeats=50, block_len=7, random_state=42
):
    """
    Blockierte, gruppenweise Permutation Importance für Zeitreihen.

    Im Gegensatz zur featureweisen Permutation werden hier **alle Features einer Quelle/Gruppe
    gleichzeitig** permutiert – und zwar **blockweise** (z. B. 7-Tage-Blöcke). Dadurch bleiben
    kurzfristige Dynamiken (Autokorrelation, Wochenrhythmik) erhalten, während die **zeitliche
    Ausrichtung** der gesamten Gruppe relativ zum Ziel y gezielt zerstört wird.
    Das liefert eine realistischere Nullhypothese für Zeitreihen und ergibt direkt
    die Importanz **auf Quellen-/Gruppenebene** (HRV, Bewegung, Sprache, …).

    Parameters
    ----------
    fitted_estimator : sklearn estimator (oder Pipeline)
        Bereits **auf dem Trainingssplit** gefittetes Modell mit `.predict(X)`-Methode.
        (Typisch: eure Pipeline aus Preprocessing + ElasticNet.)
    X_test : pandas.DataFrame
        Test-Featurematrix (chronologisch sortiert). Muss Spaltennamen tragen, die in `groups` vorkommen können.
    y_test : array-like
        Zielwerte für den Testsatz (gleiche Länge wie X_test).
    groups : dict[str, list[str]]
        Mapping {gruppenname: [feature_spaltennamen, ...]}.
        Nicht vorhandene Spalten werden still ignoriert (z. B. wenn ein Feature in diesem Fold gefiltert wurde).
    n_repeats : int, default=50
        Anzahl Wiederholungen pro Gruppe für die Monte-Carlo-Schätzung der Importanz.
    block_len : int, default=7
        Länge der Zeitblöcke für die Block-Permutation (z. B. 7 für Wochenrhythmik
        oder die erste Lag-Länge, bei der die ACF ~ 0 ist).
    random_state : int | None, default=42
        Seed für Reproduzierbarkeit (numpy Generator).

    Returns
    -------
    pandas.DataFrame
        Tabelle mit einer Zeile pro Gruppe und folgenden Spalten:
        - 'group'            : Name der Quelle/Gruppe
        - 'delta_R2_mean'    : mittlerer Abfall in R² (baseline_R2 - R2_perm) über n_repeats
        - 'delta_R2_std'     : Standardabweichung des R²-Abfalls
        - 'delta_MAE_mean'   : mittlere Zunahme in MAE (MAE_perm - baseline_MAE)
        - 'delta_MAE_std'    : Standardabweichung der MAE-Differenz
        - 'n_repeats'        : Anzahl Wiederholungen
        - 'block_len'        : verwendete Blocklänge

    Notes
    -----
    - **Warum gruppenweise?** Viele eurer Features innerhalb einer Quelle sind korreliert.
      Permutiert man nur ein einzelnes Feature, kann die Quelle weiterhin über korrelierte
      Geschwisterinformationen wirken. Die gruppierte Permutation testet den *gemeinsamen*
      Beitrag der Quelle.
    - **Warum blockweise?** i.i.d.-Permutation zerstört Autokorrelation/Seasonality
      und überschätzt Importanz. Block-Permutation hält kurzfristige Dynamik intakt
      und prüft die richtige Nullhypothese „Quelle vorhanden, aber zeitlich falsch ausgerichtet“.
    - **Skalierung:** Die Funktion setzt voraus, dass `fitted_estimator` alle nötigen
      Preprocessing-Schritte enthält (z. B. Scaler in Pipeline). X_test muss bereits im
      Testraum vorliegen (d. h. dieselbe Spaltenordnung/ -namen wie im Fit).
    - **Relativierung auf 100 %:** Für 100 %-Plots pro Fold (Quellenanteile) normalisiere die
      resultierenden delta-R² je Fold auf Summe = 1, bevor man über Folds/Probanden mittelt.

    Examples
    --------
    >>> groups = {
    ...     "HRV": ["HRV_ULF", "HRV_SDNN"],
    ...     "ACC": ["steps", "wake intensity gradient"],
    ...     "DailyDialog": ["F0semitoneFrom27", "jitterLocal_sma3nz_amean"]
    ... }
    >>> df_pi_g = permutation_importance_by_group(model, X_test, y_test, groups,
    ...                                           n_repeats=50, block_len=7, random_state=42)
    >>> df_pi_g.sort_values("delta_R2_mean", ascending=False).head()

    """
    rng = np.random.default_rng(random_state)
    y_base = fitted_estimator.predict(X_test)
    R2_base = r2_score(y_test, y_base)
    MAE_base = mean_absolute_error(y_test, y_base)

    rows = []
    for gname, gcols in groups.items():
        col_idx = [X_test.columns.get_loc(c) for c in gcols if c in X_test.columns]
        if not col_idx: 
            continue
        dR2_list, dMAE_list = [], []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            # alle Spalten der Gruppe gemeinsam block-permutieren (gleicher Blockorder)
            n = len(Xp)
            blocks = [np.arange(i, min(i+block_len, n)) for i in range(0, n, block_len)]
            order = rng.permutation(len(blocks))
            new_idx = np.concatenate([blocks[i] for i in order])
            for j in col_idx:
                Xp.iloc[:, j] = Xp.iloc[new_idx, j].to_numpy()

            y_hat = fitted_estimator.predict(Xp)
            dR2_list.append(R2_base - r2_score(y_test, y_hat))
            dMAE_list.append(mean_absolute_error(y_test, y_hat) - MAE_base)
        rows.append({"group": gname, "delta_R2": np.mean(dR2_list), "delta_MAE": np.mean(dMAE_list)})
    return pd.DataFrame(rows).sort_values("delta_R2", ascending=False).reset_index(drop=True)


def _block_permute_col(X_df, col_idx, block_len, rng):
    """
    Blockweise Permutation einer Spalte in zeitlich geordneten Testdaten.

    Diese Funktion permutiert eine einzelne Feature-Spalte nicht punktweise (i.i.d.),
    sondern in zusammenhängenden Zeitblöcken fester Länge. Dadurch bleiben
    innerhalb eines Blocks kurzfristige Dynamik (Autokorrelation, Wochenrhythmik)
    erhalten, während die zeitliche Ausrichtung der Blöcke relativ zum Ziel y
    gezielt zerstört wird. Das ist für Zeitreihen-Permutation-Importance die
    realistischere Nullhypothese („Feature vorhanden, aber falsch ausgerichtet“).

    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature-Matrix des (chronologisch sortierten) Test-Splits.
        Jede Zeile entspricht typischerweise einem Tag.
    col_idx : int
        Spaltenindex (0-basiert) der zu permutierenden Feature-Spalte in X_df.
        (Nutze X_df.columns.get_loc(name), wenn du einen Spaltennamen hast.)
    block_len : int
        Blocklänge (Anzahl aufeinanderfolgender Zeilen), die als Einheit
        zusammen permutiert werden. Typische Wahl: 7 (Wochenrhythmus) oder
        die erste Lag-Länge, bei der die ACF ~ 0 ist.
    rng : numpy.random.Generator
        Zufallsgenerator für reproduzierbare Permutation (z. B. np.random.default_rng(42)).

    Returns
    -------
    pandas.DataFrame
        Kopie von X_df, in der genau die gewählte Spalte blockweise permutiert wurde.
        Alle anderen Spalten bleiben unverändert.

    Notes
    -----
    - Zu kleine Blocklängen nähern die i.i.d.-Permutation an und können die
      Autokorrelation zu stark zerstören (Überschätzung der Importance).
    - Zu große Blocklängen entkoppeln ggf. zu wenig (Unterschätzung).
      Validiere 7/14 gegen die ACF deiner Reihe.
    - Für korrelierte Features derselben Quelle (z. B. HRV-Cluster) sollten
      alle betroffenen Spalten gemeinsam blockweise permutiert werden
      (grouped permutation), um Quellstruktur zu erhalten.

    Example
    -------
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": range(10), "b": np.linspace(0, 1, 10)})
    X_perm = _block_permute_col(X, col_idx=0, block_len=4, rng=rng)
    len(X_perm) == len(X)
    True
    sorted(X_perm["a"].tolist()) == list(range(10))  # Werte erhalten, Reihenfolge verändert
    True  
    """
    Xp = X_df.copy(deep=True)
    n = len(Xp)
    # Indizes in Blöcke partitionieren
    blocks = [np.arange(i, min(i+block_len, n)) for i in range(0, n, block_len)]
    order = rng.permutation(len(blocks))
    new_idx = np.concatenate([blocks[i] for i in order])
    Xp.iloc[:, col_idx] = Xp.iloc[new_idx, col_idx].to_numpy()
    return Xp

def _fold_normalize(
    df: pd.DataFrame,
    fold_col: str,
    value_col: str,
    rel_col: str = "rel_value"
) -> pd.DataFrame:
    """
    Normiert Importanzwerte pro Fold so, dass die Summe je Fold = 1 ist.

    Für 100%-Darstellungen und robuste Aggregation über Folds werden die innerhalb
    eines Folds berechneten Importanzen (z. B. mean_abs_shap oder delta_R2) auf die
    fold-spezifische Gesamtsumme skaliert. Dadurch sind Features/Quellen zwischen
    Folds vergleichbar; anschließende Median-/IQR-Aggregation reflektiert stabile
    relative Bedeutung statt absoluter Skalenartefakte.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabelle mit mindestens den Spalten (fold_col, value_col).
    fold_col : str
        Spaltenname des Fold-Identifiers (z. B. "fold").
    value_col : str
        Spaltenname des zu normierenden Werts (z. B. "mean_abs_shap" oder "delta_R2").
    rel_col : str, optional
        Name der neuen Spalte für den relativen, auf 1 normierten Wert.

    Returns
    -------
    pandas.DataFrame
        Kopie von df mit zusätzlich der Spalte `rel_col`.

    Notes
    -----
    - Falls die Fold-Summe 0 ist, wird der relative Wert auf 0 gesetzt.
    - Für Gruppen-PI kann delta_R2 bevorzugt normiert werden (Interpretation als
      erklärter Varianzbeitrag); analog ist eine MAE-Normierung möglich.

    Example
    -------
    >>> shap_folds_df = _fold_normalize(shap_folds_df, "fold", "mean_abs_shap", "rel_mean_abs_shap")
    >>> pi_folds_df   = _fold_normalize(pi_folds_df, "fold", "delta_R2",      "rel_delta_R2")
    """
    out = df.copy()
    sums = out.groupby(fold_col)[value_col].transform("sum").replace(0, np.nan)
    out[rel_col] = (out[value_col] / sums).fillna(0.0)
    return out

def run_explain_over_splits(
    X_df: pd.DataFrame,
    y: np.ndarray,
    make_model,                    # callable -> unfit model (EN/RF/...)
    shap_kind: str = "auto",       # "linear", "tree", "auto"
    compute_shap: bool = True,
    compute_pi: bool = True,
    n_splits: int = 5,
    test_size: int = 30,
    embargo: int = 0,
    min_train_size: int = 90,
    n_repeats: int = 50,
    random_state: int = 42,
    fixed_scaler: StandardScaler | None = None,
):
    """
    Expanding-/Rolling-Evaluation über zeitlich geordnete Splits mit optionaler
    SHAP-Analyse und (blockierter) Permutation Importance auf dem Testfenster.

    Für eine chronologisch sortierte Tagesreihe erzeugt die Funktion Rolling-Origin-Splits
    (Expanding per Default), trainiert je Fold eine Preprocessing+Modell-Pipeline auf dem
    Trainingsfenster, evaluiert auf dem Testfenster (R²/MAE) und berechnet optional
    SHAP-Attributionen sowie blockierte Permutation Importances. Zusätzlich kann eine
    gruppenweise PI (z. B. HRV/ACT/COMM/SLEEP) berechnet werden, um Quellen-Anteile
    direkt zu erhalten (für 100%-Plots und statistische Tests).

    Parameters
    ----------
    X_df : pandas.DataFrame
        Chronologisch sortierte Feature-Matrix (Zeilen = Tage).
    y : np.ndarray
        Zielvektor gleicher Länge wie X_df.
    make_model : callable
        Fabrikfunktion, die einen **ungefitten** Estimator/Pipeline zurückgibt (z. B. ElasticNet).
    shap_kind : {"auto","linear","tree"}, optional
        SHAP-Backend-Auswahl; "auto" versucht kompatible Explainer zu wählen.
    compute_shap : bool, optional
        Ob pro Fold SHAP-Werte auf dem Testfenster berechnet werden.
    compute_pi : bool, optional
        Ob pro Fold Permutation Importances berechnet werden (Feature-weise + Gruppen-weise).
    n_splits : int, optional
        Anzahl Rolling-Origin-Folds.
    test_size : int, optional
        Länge des Testfensters je Fold (in Tagen).
    embargo : int, optional
        Sicherheitslücke zwischen Train-Ende und Test-Start (Tage), um Leakage zu vermeiden.
    min_train_size : int, optional
        Minimale Trainingsfensterlänge (in Tagen).
    n_repeats : int, optional
        Wiederholungen pro Feature/Gruppe für PI (Monte-Carlo-Schätzung).
    random_state : int, optional
        Seed für Reproduzierbarkeit (wird für Permutation verwendet).
    fixed_scaler : StandardScaler | None, optional
        Optionaler, bereits gefitteter globaler StandardScaler, der **in allen Folds
        unverändert** auf Train/Test angewandt wird (für konsistente Feature-Skalen
        und damit vergleichbare SHAP/PI über Folds). Falls None, wird kein fixer
        Scaler genutzt (d. h. Preprocessing wird nur auf Train gefittet).

    Returns
    -------
    dict
        {
          "fold_metrics_df":  DataFrame mit R²/MAE je Fold,
          "shap_folds_df":    DataFrame mit (fold, feature, mean_abs_shap, mean_signed_shap) oder None,
          "shap_summary_df":  Median-aggregation über Folds oder None,
          "pi_folds_df":      DataFrame mit (fold, feature, delta_R2, delta_MAE) oder None,
          "pi_summary_df":    Median-aggregation über Folds oder None,
          "pi_groups_folds_df":   DataFrame mit (fold, group, delta_R2_mean, delta_MAE_mean, rel_*) oder None,
          "pi_groups_summary_df": Median-aggregation über Folds oder None,
        }

    Notes
    -----
    - Preprocessing wird **ausschließlich auf TRAIN** gefittet und auf TEST angewandt (leakage-sicher).
    - Permutation erfolgt **blockweise** (CONFIG["pi"]["block_len"]), um Autokorrelation/Seasonality zu respektieren.
    - Gruppen-PI erwartet CONFIG["feature_groups"] als Mapping {gruppe: [features]} (z. B. COARSE_GROUPS oder DAY_NIGHT_GROUPS).
    - Für 100%-Plots: pro Fold die Gruppen-Importanzen relativieren (Summe = 1), dann über Folds/Personen mitteln (Median + IQR).
    - SHAP/PI werden für 100%-Darstellungen fold-normalisiert (Summe je Fold = 1):
        SHAP: rel_mean_abs_shap; PI: rel_delta_R2. 
        Die Summary-Tabellen basieren auf Medianen dieser relativen Werte, was robuste, vergleichbare Aggregation
        über Folds (und später über Personen) erlaubt.

    Example
    -------
    >>> CONFIG["pi"] = {"block_len": 7, "n_repeats": 50}
    >>> CONFIG["feature_groups"] = COARSE_GROUPS   # oder DAY_NIGHT_GROUPS
    >>> res = run_explain_over_splits(
    ...     X_df, y, make_model=make_en_pipeline, compute_shap=True, compute_pi=True,
    ...     n_splits=5, test_size=14, min_train_size=90, random_state=42
    ... )
    >>> res["pi_groups_summary_df"].head()
    """

    splits = rolling_origin_splits(
        n_samples=len(y),
        n_splits=n_splits,
        test_size=test_size,
        embargo=embargo,
        min_train_size=min_train_size,
    )

    rng = np.random.default_rng(random_state)
    fold_rows = []
    shap_rows = []
    pi_rows = []
    pi_group_rows = []

    for fold_id, (tr, te) in enumerate(splits, start=1):
        pp = preprocess_train_only()                          # <<< statt preprocess_pipeline
        Xt_tr = pp.fit_transform(X_df.iloc[tr])               # train-only fit
        Xt_te = pp.transform(X_df.iloc[te])

        if fixed_scaler is not None:                          # <<< Fix-Scaler anwenden (nur transform)
            Xt_tr = fixed_scaler.transform(Xt_tr)
            Xt_te = fixed_scaler.transform(Xt_te)

        # --- Modell fitten
        model = make_model()
        model.fit(Xt_tr, y[tr])

        # --- Test-Metriken
        y_hat = model.predict(Xt_te)
        R2 = r2_score(y[te], y_hat)
        MAE = mean_absolute_error(y[te], y_hat)
        fold_rows.append({
            "fold": fold_id,
            "r2": float(R2),
            "mae": float(MAE),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_end_idx": int(tr[-1]),
            "test_start_idx": int(te[0]),
            "test_end_idx": int(te[-1]),
        })

        # --- SHAP
        if compute_shap:
            try:
                if shap_kind == "linear":
                    explainer = shap.LinearExplainer(model, Xt_tr)
                    shap_vals = explainer.shap_values(Xt_te)
                elif shap_kind == "tree":
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(Xt_te)
                else:
                    expl = shap.Explainer(model, Xt_tr)
                    expl_out = expl(Xt_te)
                    shap_vals = getattr(expl_out, "values", expl_out)
            except Exception:
                expl = shap.Explainer(model, Xt_tr)
                expl_out = expl(Xt_te)
                shap_vals = getattr(expl_out, "values", expl_out)

            cols = [str(c).split("__")[-1] for c in list(Xt_te.columns)]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            mean_signed = np.asarray(shap_vals).mean(axis=0)
            for f, a, s in zip(cols, mean_abs, mean_signed):
                shap_rows.append({
                    "fold": fold_id,
                    "feature": f,
                    "mean_abs_shap": float(a),
                    "mean_signed_shap": float(s),
                })

        # --- Permutation Importance – auf Testfenster pro Block pro Feture
        if compute_pi:
            cols = list(Xt_te.columns)
            # Baseline für deltas
            y_hat_base = y_hat
            R2_base = R2
            MAE_base = MAE
            for j, col in enumerate(cols):
                dR2_list, dMAE_list = [], []
                for _ in range(n_repeats):
                    Xp = _block_permute_col(Xt_te, j, block_len=CONFIG["pi"]["block_len"], rng=rng) 
                    y_hat_p = model.predict(Xp)
                    dR2_list.append(R2_base - r2_score(y[te], y_hat_p))
                    dMAE_list.append(mean_absolute_error(y[te], y_hat_p) - MAE_base)
                pi_rows.append({
                    "fold": fold_id,
                    "feature": str(col).split("__")[-1],
                    "delta_R2": float(np.mean(dR2_list)),
                    "delta_MAE": float(np.mean(dMAE_list)),
                })
        # --- Gruppenweise Permutation Importance (blockweise)
      
        df_pi_groups = permutation_importance_by_group(
            fitted_estimator=model,
            X_test=Xt_te,
            y_test=y[te],
            groups= CONFIG["feature_groups"], # TODO: into fun args?
            n_repeats=CONFIG["pi"]["n_repeats"],
            block_len=CONFIG["pi"]["block_len"],
            random_state=CONFIG["random_state"],
        )

        # Relativierung auf 100 % (delta_R2)
        total = df_pi_groups["delta_R2"].sum()
        if total > 0:
            df_pi_groups["rel_delta_R2"] = df_pi_groups["delta_R2"] / total
        else:
            df_pi_groups["rel_delta_R2"] = 0.0

        # (delta_MAE)
        total = df_pi_groups["delta_MAE"].sum()
        if total > 0:
            df_pi_groups["rel_delta_MAE"] = df_pi_groups["delta_MAE"] / total
        else:
            df_pi_groups["rel_delta_MAE"] = 0.0
        pi_group_rows.extend(df_pi_groups.to_dict(orient="records"))

    # Ergebnisse zusammenfassen
    fold_metrics_df = pd.DataFrame(fold_rows)

    shap_folds_df = pd.DataFrame(shap_rows) if compute_shap else None
    shap_summary_df = (shap_folds_df.groupby("feature")[["mean_abs_shap","mean_signed_shap"]]
                       .median().sort_values("mean_abs_shap", ascending=False).reset_index()
                       ) if compute_shap and not shap_folds_df.empty else None

    if compute_shap and shap_folds_df is not None and not shap_folds_df.empty:
        shap_folds_df = _fold_normalize(
            shap_folds_df, fold_col="fold", value_col="mean_abs_shap", rel_col="rel_mean_abs_shap"
        )
        shap_summary_df = (
            shap_folds_df.groupby("feature")[["rel_mean_abs_shap", "mean_signed_shap"]]
            .median()
            .sort_values("rel_mean_abs_shap", ascending=False)
            .reset_index()
        )
    else:
        shap_summary_df = None

    pi_folds_df = pd.DataFrame(pi_rows) if compute_pi else None
    pi_summary_df = (pi_folds_df.groupby("feature")[["delta_R2","delta_MAE"]]
                     .median().sort_values("delta_R2", ascending=False).reset_index()
                     ) if compute_pi and not pi_folds_df.empty else None
    
    if compute_pi and pi_folds_df is not None and not pi_folds_df.empty:
        pi_folds_df = _fold_normalize(
            pi_folds_df, fold_col="fold", value_col="delta_R2", rel_col="rel_delta_R2"
        )
        pi_summary_df = (
            pi_folds_df.groupby("feature")[["rel_delta_R2", "delta_MAE"]]
            .median()
            .sort_values("rel_delta_R2", ascending=False)
            .reset_index()
        )
    else:
        pi_summary_df = None

    pi_groups_folds_df = pd.DataFrame(pi_group_rows) if compute_pi else None
    pi_groups_summary_df = (pi_groups_folds_df.groupby("group")[["delta_R2","delta_MAE","rel_delta_R2"]]
                            .median().sort_values("delta_R2", ascending=False).reset_index()
                            ) if compute_pi and pi_groups_folds_df is not None and not pi_groups_folds_df.empty else None
    
    if compute_pi and pi_groups_folds_df is not None and not pi_groups_folds_df.empty:
        pi_groups_summary_df = (
            pi_groups_folds_df.groupby("group")[["rel_delta_R2", "delta_R2", "delta_MAE"]]
            .median()
            .sort_values("rel_delta_R2", ascending=False)
            .reset_index()
        )
    else:
        pi_groups_summary_df = None

    return {
        "fold_metrics_df": fold_metrics_df,
        "shap_folds_df": shap_folds_df,
        "shap_summary_df": shap_summary_df,
        "pi_folds_df": pi_folds_df,
        "pi_summary_df": pi_summary_df,
        "pi_groups_folds_df": pi_groups_folds_df,
        "pi_groups_summary_df": pi_groups_summary_df, 
    }

# Save-Helper für die Ergebnisse aus run_explain_over_splits
def save_explain_outputs(res_dict: dict, results_dir: str, pseudonym: str, model_tag: str):
    os.makedirs(results_dir, exist_ok=True)

    # Folds-Metriken
    fm = res_dict.get("fold_metrics_df", None)
    if fm is not None and not fm.empty:
        fm.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_fold_metrics.csv"), index=False)

    # SHAP
    sf = res_dict.get("shap_folds_df", None)
    ss = res_dict.get("shap_summary_df", None)
    if sf is not None and not sf.empty:
        sf.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_shap_folds.csv"), index=False)
    if ss is not None and not ss.empty:
        ss.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_shap_summary.csv"), index=False)

    # Permutation Importance
    pf = res_dict.get("pi_folds_df", None)
    ps = res_dict.get("pi_summary_df", None)
    if pf is not None and not pf.empty:
        pf.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_feat_folds.csv"), index=False)
    if ps is not None and not ps.empty:
        ps.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_feat_summary.csv"), index=False)

    # Grouped Permutation Importance
    pgf = res_dict.get("pi_groups_folds_df", None)
    pgs = res_dict.get("pi_groups_summary_df", None)
    if pgf is not None and not pgf.empty:
        pgf.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_group_folds.csv"), index=False)
    if pgs is not None and not pgs.empty:
        pgs.to_csv(os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_group_summary.csv"), index=False)

def days_since_start(group):
    start_date = group["timestamp_utc"].min()
    return (group["timestamp_utc"] - start_date).dt.days

# %% Main Processing Function ===================================================
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

        # --- Dummy Regressor ---
        test_frac = CONFIG["init_frac_for_hp"]
        gap = 0
        split = int(round((1 - test_frac) * n))
        idx_train = np.arange(0, split - gap)
        idx_test = np.arange(split, n)

        X_train_df = X.iloc[idx_train].copy()
        X_test_df = X.iloc[idx_test].copy()
        y_train = y[idx_train]
        y_test = y[idx_test]

        print(f"{test_frac}:var(y_train) =", float(np.var(y_train)))
        print(f"{1-test_frac}:var(y_test) =", float(np.var(y_test)))
        dum = DummyRegressor(strategy="mean").fit(X_train_df, y_train)
        print("baseline R2 =", r2_score(y_test, dum.predict(X_test_df)))

        # Startfenster, Fix-Scaler fitten & EN-Hyperparameter bestimmen
        init_end = split_initial_window(n_samples=n,                          
                                        frac=CONFIG["init_frac_for_hp"],
                                        min_train_size=CONFIG["shap"]["min_train_size"])

        # Train-only Preprocessing (Imputation, per-Feature-Scaler) auf Startfenster
        pp_train_only = preprocess_train_only()
        X_init_tr = pp_train_only.fit_transform(X.iloc[:init_end])

        # Optional: globalen StandardScaler nur im Startfenster fitten und einfrieren
        fixed_scaler = None
        if CONFIG["use_fixed_scaler"]:
            fixed_scaler = fit_fixed_global_scaler(X_init_tr)
            X_init_for_cv = fixed_scaler.transform(X_init_tr)
        else:
            X_init_for_cv = X_init_tr

        en_cv = ElasticNetCV(
            l1_ratio=CONFIG["en"]["l1_ratio"],
            alphas=np.logspace(-3, 2, 50),
            cv=TimeSeriesSplit(n_splits=CONFIG["cv_n_splits"]),
            max_iter=200_000,
            tol=1e-2,
            random_state=RANDOM_STATE,
        )
        en_cv.fit(X_init_for_cv, y[:init_end])
        best_alpha = float(en_cv.alpha_)
        best_l1    = float(getattr(en_cv, "l1_ratio_", CONFIG["en"]["l1_ratio"]))
        print(f"[Init-HP] EN alpha={best_alpha:.4g}, l1_ratio={best_l1}")


        # =================== ELASTIC NET: (fixe HP aus Startfenster) + SHAP ====================
        print("[EN] Fit mit fixen HP (aus Startfenster) ...")

        # --- PREPROCESSING: train-only pro Split fitten, danach optional Fix-Scaler anwenden
        pp_en = preprocess_train_only()
        Xt_train_en = pp_en.fit_transform(X_train_df)
        Xt_test_en  = pp_en.transform(X_test_df)
        if fixed_scaler is not None:
            Xt_train_en = fixed_scaler.transform(Xt_train_en)
            Xt_test_en  = fixed_scaler.transform(Xt_test_en)

        # --- MODELL: ElasticNet mit fixen HP aus dem Startfenster
        model_en = ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1,
            max_iter=CONFIG["en"]["max_iter"],
            tol=CONFIG["en"]["tol"],
            random_state=RANDOM_STATE,
        )
        model_en.fit(Xt_train_en, y_train)

        # --- Holdout-Performance (Prospektiv)
        y_pred_en = model_en.predict(Xt_test_en)
        r2_elastic = round(r2_score(y_test, y_pred_en), 2)
        mae_elastic = round(mean_absolute_error(y_test, y_pred_en), 1)

        # --- SHAP auf preprocessed Matrizen
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

        # Feature-Namen (falls DataFrame-DesignMatrix)
        en_feat_names = [str(c).split('__')[-1] for c in list(getattr(X_train_df, 'columns', []))] or [f"f{i}" for i in range(Xt_test_en.shape[1])]
        plot_shap_summary_and_bar(RESULTS_DIR, en_shap_values, Xt_test_en, en_feat_names, f"{pseudonym}_EN")

        # Modelle speichern
        joblib.dump({"pre": pp_en, "scaler": fixed_scaler, "model": model_en},
                    os.path.join(RESULTS_DIR, "models", f"{pseudonym}_elasticnet.joblib"))

        # ================= RANDOM FOREST: GridSearch NUR im Startfenster (+ fix) ====================
        print("[RF] GridSearchCV (TimeSeriesSplit) NUR Startfenster ...")

        rf_pre = preprocess_train_only()
        X_rf_tr = rf_pre.fit_transform(X.iloc[:init_end])
        if fixed_scaler is not None:
            X_rf_tr = fixed_scaler.transform(X_rf_tr)

        rf_search_est = Pipeline([
            ("model", RandomForestRegressor(random_state=RANDOM_STATE))
        ])
        rf_gs = GridSearchCV(
            estimator=rf_search_est,
            param_grid=CONFIG["rf"]["param_grid"],
            scoring=CONFIG["rf"]["scoring"],
            cv=TimeSeriesSplit(n_splits=CONFIG["cv_n_splits"]),
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        rf_gs.fit(X_rf_tr, y[:init_end])
        rf_best_params = rf_gs.best_params_
        print("[Init-HP] RF best:", rf_best_params)

        rf_best_params_plain = {
            k.split("__", 1)[1]: v for k, v in rf_best_params.items() if k.startswith("model__")
        }
        
        rf_pre_fold = preprocess_train_only()
        Xt_train_rf = rf_pre_fold.fit_transform(X_train_df)
        Xt_test_rf  = rf_pre_fold.transform(X_test_df)
        if fixed_scaler is not None:
            Xt_train_rf = fixed_scaler.transform(Xt_train_rf)
            Xt_test_rf  = fixed_scaler.transform(Xt_test_rf)

        model_rf = RandomForestRegressor(random_state=RANDOM_STATE, **rf_best_params_plain)
        model_rf.fit(Xt_train_rf, y_train)

        y_pred_rf = model_rf.predict(Xt_test_rf)
        r2_rf = round(r2_score(y_test, y_pred_rf), 2)
        mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 1)

        rf_explainer = shap.TreeExplainer(model_rf)
        rf_shap_values = rf_explainer.shap_values(Xt_test_rf)
        rf_feat_names = [str(c).split("__")[-1] for c in list(getattr(X_train_df, 'columns', []))] or [f"f{i}" for i in range(Xt_test_rf.shape[1])]
        plot_shap_summary_and_bar(RESULTS_DIR, rf_shap_values, Xt_test_rf, rf_feat_names, f"{pseudonym}_RF")

        print(f"RF R2={r2_rf}, MAE={mae_rf}, n_estimators={model_rf.n_estimators}, max_depth={model_rf.max_depth}, min_samples_leaf={model_rf.min_samples_leaf}")
        print("mean(|SHAP|):", float(np.nanmean(np.abs(rf_shap_values))))
        # Modelle speichern
        joblib.dump({"pre": rf_pre_fold, "scaler": fixed_scaler, "model": model_rf},
                    os.path.join(RESULTS_DIR, "models", f"{pseudonym}_rf.joblib"))

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
            "elastic": {"pred": list(model_en.predict(fixed_scaler.transform(pp_en.transform(X)) if fixed_scaler else pp_en.transform(X))), "lower": None, "upper": None},
            "rf": {"pred": list(model_rf.predict(fixed_scaler.transform(rf_pre_fold.transform(X)) if fixed_scaler else rf_pre_fold.transform(X))), "lower": None, "upper": None},
        }

        # ==================== Ergebnisse sammeln für ini Fold===============================
        results[pseudonym] = {
            "r2_elastic": r2_elastic,
            "mae_elastic": mae_elastic,
            "r2_rf": r2_rf,
            "mae_rf": mae_rf,
            "rmssd_phq2": rmssd_phq2,
            "en_best_alpha": best_alpha,
            "en_best_l1_ratio": best_l1,
            "rf_best_n_estimators": model_rf.n_estimators,
            "rf_best_max_depth": model_rf.max_depth,
            "rf_best_min_samples_leaf": model_rf.min_samples_leaf,
        }

        # ==================== Rolling PI & Rolling SHAP ========================
        print("  [EN] Rolling PI/SHAP ...")

        make_en = lambda: ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1,
            max_iter=CONFIG["en"]["max_iter"],
            tol=CONFIG["en"]["tol"],
            random_state=RANDOM_STATE,
        )

        res_en = run_explain_over_splits(
            X_df=X, y=y,
            make_model=make_en,
            shap_kind="linear",
            compute_shap=True, compute_pi=True,
            n_splits=CONFIG["shap"]["n_splits"],
            test_size=CONFIG["shap"]["test_size"],
            embargo=CONFIG["shap"]["embargo"],
            min_train_size=CONFIG["shap"]["min_train_size"],
            n_repeats=CONFIG["pi"]["n_repeats"],
            random_state=CONFIG["random_state"],
            fixed_scaler=fixed_scaler,
        )


        print("  [RF] Rolling PI/SHAP ...")

        def make_rf():
            return RandomForestRegressor(
                random_state=RANDOM_STATE,
                **{k: rf_best_params[k] for k in ["n_estimators","max_depth","min_samples_leaf","max_features"] if k in rf_best_params}
            )
        
        res_rf = run_explain_over_splits(
            X_df=X, y=y,
            make_model=make_rf,
            shap_kind="tree",
            compute_shap=True, compute_pi=True,
            n_splits=CONFIG["shap"]["n_splits"],
            test_size=CONFIG["shap"]["test_size"],
            embargo=CONFIG["shap"]["embargo"],
            min_train_size=CONFIG["shap"]["min_train_size"],
            n_repeats=CONFIG["pi"]["n_repeats"],
            random_state=CONFIG["random_state"],
            fixed_scaler=fixed_scaler,
        )

        # ============== Ergebnisse speichern (EN/RF) ==============
        save_explain_outputs(res_en, RESULTS_DIR, pseudonym, "EN")
        save_explain_outputs(res_rf, RESULTS_DIR, pseudonym, "RF")

        # ============== Zusammenfassungs-Plots (Top-K) ==============
        # EN – SHAP/PI Bars
        if res_en.get("shap_summary_df") is not None and not res_en["shap_summary_df"].empty:
            plot_importance_bars(
                res_en["shap_summary_df"], "rel_mean_abs_shap",
                title=f"{pseudonym} EN – SHAP (median über Folds)",
                outpath=os.path.join(RESULTS_DIR, f"{pseudonym}_EN_shap_summary_bar.png"),
                top_k=TOP_K
            )
        if res_en.get("pi_summary_df") is not None and not res_en["pi_summary_df"].empty:
            plot_importance_bars(
                res_en["pi_summary_df"], "rel_delta_R2",
                title=f"{pseudonym} EN – Permutation Importance ΔR² (median über Folds)",
                outpath=os.path.join(RESULTS_DIR, f"{pseudonym}_EN_pi_summary_bar.png"),
                top_k=TOP_K
            )

        # RF – SHAP/PI Bars
        if res_rf.get("shap_summary_df") is not None and not res_rf["shap_summary_df"].empty:
            plot_importance_bars(
                res_rf["shap_summary_df"], "rel_mean_abs_shap",
                title=f"{pseudonym} RF – SHAP (median über Folds)",
                outpath=os.path.join(RESULTS_DIR, f"{pseudonym}_RF_shap_summary_bar.png"),
                top_k=TOP_K
            )
        if res_rf.get("pi_summary_df") is not None and not res_rf["pi_summary_df"].empty:
            plot_importance_bars(
                res_rf["pi_summary_df"], "rel_delta_R2",
                title=f"{pseudonym} RF – Permutation Importance ΔR² (median über Folds)",
                outpath=os.path.join(RESULTS_DIR, f"{pseudonym}_RF_pi_summary_bar.png"),
                top_k=TOP_K
            )

        # ============== Folds auf PHQ-2 Zeitreihe ==============
        # timestamps passend zu X_df_valid / y_valid:
        timestamps_valid = (df_participant
                            .loc[df_participant[target_column].notna(), "timestamp_utc"]
                            .reset_index(drop=True))

        # EN-Folds
        if res_en.get("fold_metrics_df") is not None and not res_en["fold_metrics_df"].empty:
            plot_folds_on_timeseries(
                timestamps=timestamps_valid,
                y_values=y,
                fold_metrics_df=res_en["fold_metrics_df"],
                out_path=os.path.join(RESULTS_DIR, f"{pseudonym}_EN_folds_timeseries.png"),
                title=f"{pseudonym} – EN: Test-Fenster & Metriken"
            )

        # RF-Folds
        if res_rf.get("fold_metrics_df") is not None and not res_rf["fold_metrics_df"].empty:
            plot_folds_on_timeseries(
                timestamps=timestamps_valid,
                y_values=y,
                fold_metrics_df=res_rf["fold_metrics_df"],
                out_path=os.path.join(RESULTS_DIR, f"{pseudonym}_RF_folds_timeseries.png"),
                title=f"{pseudonym} – RF: Test-Fenster & Metriken"
            )

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

# %% Caching mechanism ===========================================================
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
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "elastic", show_pred_ci=False)
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "elastic", show_pred_ci=False)

plot_phq2_timeseries_from_results(RESULTS_DIR, plot_data, "rf")
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf", show_pred_ci=False)
plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf", show_pred_ci=False)

# %% Save evaluation metrics (per participant) ==================================
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv(f"{RESULTS_DIR}/model_performance_summary.csv")

# %%
