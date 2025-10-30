# %% Data Loading and Setup =====================================================
import json
import os
from datetime import timedelta
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from RepeatedTimeSeriesSplit import RepeatedTimeSeriesSplit

set_config(transform_output="pandas")

# ==== RUN/PROJECT CONFIG ======================================================
CONFIG = {
    "random_state": 42,
    "top_k": 15,
    "paths": {
        "data": "data/df_merged_dupsfree_v8.pickle",
        "results_dir": "results_dummycomparison_timeaware_5050_split",
    },
    "targets": {
        "phq2": "abend_PHQ2_sum",
        "phq9": "woche_PHQ9_sum",
    },
    # Anteil Startfenster für HP-Tuning und Pipeline-Fit
    "init_frac_for_hp": 0.50,
    # GridSearchCV (TimeSeriesSplit)
    "cv_n_splits": 5,
    # Gap/Embargo in TimeSeriesSplit für CV
    "cv_embargo": 1,
    # Elastic Net
    "en": {
        "scoring": "r2",  # "neg_mean_absolute_error" # MAE vs. r2:  r2 "bestraft" mehr konstante verläufe.
        "tune_l1_ratio": False,  # True: l1_ratio in [0.1, 0.5, 0.9] mit-tunen
        "max_iter": 100000,
        "tol": 1e-3,
        "y_min_var_init": 1e-3,  # min. Varianz im Startfenster (sonst erweitern)
        "param_grid": {
            "model__alpha": list(np.logspace(-6, 1, 80)),
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
    },
    # Random Forest
    "rf": {
        "scoring": "r2",
        "param_grid": {
            "model__max_depth": [4, 6, 8, 10, 15, 20, None],
            "model__n_estimators": [100, 200, 400, 800],
            "model__min_samples_leaf": [1, 2, 4, 6, 8],
        },
    },
    # Rolling Permutation Importance
    "pi": {
        "test_size": 30,
        "train_size": 30,
        "embargo": 0,
        "n_repeats": 10,  # 10 nunr für smoke test (sonst 50)
        "agg": "mean",  # "mean" oder "median"
        "block_len": 7,  # Länge der Blöcke für blockweise Permutation
    },
}


# Ordner und Cachepfad
os.makedirs(CONFIG["paths"]["results_dir"], exist_ok=True)
CONFIG["paths"]["cache_results"] = os.path.join(
    CONFIG["paths"]["results_dir"], "process_participants_results.pkl"
)
with open(
    os.path.join(CONFIG["paths"]["results_dir"], "config.json"), "w", encoding="utf-8"
) as f:
    json.dump(CONFIG, f, indent=2)

# DEFS
DATA_PATH = CONFIG["paths"]["data"]
RESULTS_DIR = CONFIG["paths"]["results_dir"]
LOAD_CACHED_RESULTS = False
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
    "HRV": _feats(["HRV"]),
    "ACR": _feats(["steps", "ACR"]),  # Aktivität/Transitions
    "COMM": _feats(["app usage", "calls"]),  # Kommunikation / App-Use
    "AUDIO": _feats(["audio"]),  # Sprach-/Stimm-Proxys
    "SLEEP": _feats(["sleep"]),
    "CAL": _feats(["time"]),  # Kalender/Trend
}

# Day/Night-Variante für Quellen
DAY_NIGHT_GROUPS = {}
for name, cats in {
    "HRV": ["HRV"],
    "ACR": ["steps", "ACR"],
    "COMM": ["app usage", "calls"],
    "SLEEP": ["sleep"],
}.items():
    for period in ["day", "night"]:
        mask = feature_config["category"].isin(cats) & (feature_config[period] == 1)
        DAY_NIGHT_GROUPS[f"{name}_{period}"] = feature_config.loc[
            mask, "feature"
        ].tolist()

# leere Gruppen entfernen
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
    plot_elasticnet_vs_dummyregressor,
    plot_folds_on_timeseries,
    plot_importance_bars,
    plot_mae_rmssd_bar_2,
    plot_phq2_test_errors_from_results,
    plot_phq2_timeseries_from_results,
    plot_phq2_timeseries_with_adherence_from_results,
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
    transformers=[
        (f"scale_{col}", scaler, [col]) for col, scaler in feature_scalers.items()
    ],
    remainder=StandardScaler(),
    verbose_feature_names_out=False,
)


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
    test_size: int = 30,
    hp_train_frac: float = 0.3,  # fraction of samples used for initial training window
    embargo: int = 0,
    step_train: int = 30,  # extend training window by 30 each split
    min_test_size: int = 5,  # soft minimum; final split may be shorter
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding Window with flexible number of splits.
    The training window starts as a fraction (hp_train_frac) of the total samples
    and grows by `step_train` per split.
    The last split's test window always ends at n_samples.

    Fold k:
      train[0 : T_k] -> test[T_k + embargo : min(T_k + embargo + test_size, n_samples)]
    """

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    # --- Compute base training size from fraction ---
    base_train_size = int(round(hp_train_frac * n_samples))
    if base_train_size < 1:
        raise ValueError("hp_train_frac too small, results in empty training window.")
    if n_samples <= base_train_size:
        return splits

    tr_end = base_train_size

    while True:
        start_test = tr_end + embargo
        if start_test >= n_samples:
            break  # no room left for test samples

        # Cap test end at n_samples
        end_test = min(start_test + test_size, n_samples)
        is_last = end_test == n_samples

        tr_idx = np.arange(0, tr_end)
        te_idx = np.arange(start_test, end_test)

        # Add valid split
        if len(te_idx) >= min_test_size or (is_last and len(te_idx) > 0):
            splits.append((tr_idx, te_idx))

        if is_last:
            break

        # Grow training window for next iteration
        tr_end += step_train

        # Safety: if there’s not enough space left for another test fold
        if tr_end + embargo >= n_samples:
            # Add a final split that ends at n_samples if possible
            start_test = min(tr_end + embargo, n_samples - 1)
            if start_test < n_samples:
                end_test = n_samples
                tr_idx = np.arange(0, min(tr_end, n_samples))
                te_idx = np.arange(start_test, end_test)
                if len(te_idx) > 0:
                    splits.append((tr_idx, te_idx))
            break

    return splits


def preprocess_pipeline():
    """
    Preprocessing-Schritte, die pro Fold auf dem Trainingsfenster gefittet werden.

    Enthält alle Schritte, die drift-/missingness-sensitiv sind (z. B. MICE-Imputation,
    per-Feature-Transformationen je Quelle). Diese werden in jedem Fold auf dem
    Trainingsblock gefittet und dann auf Train/Test angewandt – leakage-sicher.

    Returns
    -------
    sklearn.Pipeline
        Pipeline ohne den globalen, fixen StandardScaler.

    """
    # Beispiel: baue hier deine bisherigen Steps OHNE finalen globalen StandardScaler
    # (Passe an deine realen Step-Namen/Objekte an!)
    return Pipeline(
        [
            (
                "imputer_mice",
                IterativeImputer(
                    max_iter=50,
                    tol=1e-2,
                    random_state=CONFIG["random_state"],
                    keep_empty_features=True,
                ),
            ),
            ("per_feature_scalers", pre_scalers),
            ("standard_scaler", StandardScaler()),
        ]
    )


# %%  PERMUTATION-IMPORTANCE (modell-agnostisch) =========================
def permutation_importance_per_feature(
    fitted_estimator: BaseEstimator,
    X_test: pd.DataFrame,  # PREPROCESSED DataFrame (Feature-Namen wichtig)
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
        rows.append(
            {
                "feature": str(col).split("__")[-1],
                "delta_R2": float(np.mean(dR2_list)),
                "delta_MAE": float(np.mean(dMAE_list)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("delta_R2", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_by_group(
    fitted_estimator,
    X_test,
    y_test,
    groups: Dict[str, list],
    n_repeats=50,
    block_len=7,
    random_state=42,
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
            blocks = [
                np.arange(i, min(i + block_len, n)) for i in range(0, n, block_len)
            ]
            order = rng.permutation(len(blocks))
            new_idx = np.concatenate([blocks[i] for i in order])
            for j in col_idx:
                Xp.iloc[:, j] = Xp.iloc[new_idx, j].to_numpy()

            y_hat = fitted_estimator.predict(Xp)
            dR2_list.append(R2_base - r2_score(y_test, y_hat))
            dMAE_list.append(mean_absolute_error(y_test, y_hat) - MAE_base)
        rows.append(
            {
                "group": gname,
                "delta_R2": np.mean(dR2_list),
                "delta_MAE": np.mean(dMAE_list),
                "rel_delta_R2": np.mean(dR2_list) / R2_base if R2_base != 0 else np.nan,
                "rel_delta_MAE": (
                    np.mean(dMAE_list) / MAE_base if MAE_base != 0 else np.nan
                ),
                "delta_R2_std": np.std(dR2_list, ddof=1),
                "delta_MAE_std": np.std(dMAE_list, ddof=1),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("delta_R2", ascending=False)
        .reset_index(drop=True)
    )


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
    blocks = [np.arange(i, min(i + block_len, n)) for i in range(0, n, block_len)]
    order = rng.permutation(len(blocks))
    new_idx = np.concatenate([blocks[i] for i in order])
    Xp.iloc[:, col_idx] = Xp.iloc[new_idx, col_idx].to_numpy()
    return Xp


def run_explain_over_splits(
    X_df: pd.DataFrame,
    y: np.ndarray,
    ts: pd.Series,  # <-- NEW: timestamps aligned with X_df/y
    make_model,  # callable -> unfit model (EN/RF/...)
    compute_pi: bool = True,
    hp_train_frac: float = 0.3,
    test_days: int = 30,  # 30-day test window
    step_days: int = 30,  # 30-day increments (train grows by this)
    embargo_days: int = 0,
    n_repeats: int = 50,
    random_state: int = 42,
    fixed_preprocess_pipeline: Pipeline | None = None,
):
    """
    Expanding, **time-based** rolling-origin evaluation.
    Windows are defined in days (not sample counts).
    """

    # Build time-based splits
    splits = rolling_time_splits_by_fraction(
        ts=ts,
        init_frac_for_hp=hp_train_frac,
        test_days=test_days,
        step_days=step_days,
        embargo_days=embargo_days,
        min_test_samples=1,
    )

    rng = np.random.default_rng(random_state)
    fold_rows, pi_rows, pi_group_rows = [], [], []

    for fold_id, (tr_idx, te_idx, meta) in enumerate(splits, start=1):
        # transform with fixed preprocessor (already fitted before)
        Xt_tr = fixed_preprocess_pipeline.transform(X_df.iloc[tr_idx])
        Xt_te = fixed_preprocess_pipeline.transform(X_df.iloc[te_idx])

        model = make_model()
        model.fit(Xt_tr, y[tr_idx])

        y_hat = model.predict(Xt_te)
        R2 = r2_score(y[te_idx], y_hat)
        MAE = mean_absolute_error(y[te_idx], y_hat)

        fold_rows.append(
            {
                "fold": fold_id,
                "r2": float(R2),
                "mae": float(MAE),
                "n_train_samples": int(len(tr_idx)),  # <-- counts
                "n_test_samples": int(len(te_idx)),  # <-- counts
                "train_from_ts": meta["train_from_ts"],  # <-- timestamps
                "train_to_ts": meta["train_to_ts"],
                "test_from_ts": meta["test_from_ts"],
                "test_to_ts": meta["test_to_ts"],
            }
        )

        if compute_pi:
            cols = list(Xt_te.columns)
            y_hat_base = y_hat
            R2_base = R2
            MAE_base = MAE

            for j, col in enumerate(cols):
                dR2_list, dMAE_list = [], []
                for _ in range(n_repeats):
                    Xp = _block_permute_col(
                        Xt_te, j, block_len=CONFIG["pi"]["block_len"], rng=rng
                    )
                    y_hat_p = model.predict(Xp)
                    dR2_list.append(R2_base - r2_score(y[te_idx], y_hat_p))
                    dMAE_list.append(mean_absolute_error(y[te_idx], y_hat_p) - MAE_base)
                pi_rows.append(
                    {
                        "fold": fold_id,
                        "feature": str(col).split("__")[-1],
                        "delta_R2": float(np.mean(dR2_list)),
                        "delta_MAE": float(np.mean(dMAE_list)),
                        "rel_delta_R2": (
                            float(np.mean(dR2_list) / R2_base)
                            if R2_base != 0
                            else np.nan
                        ),
                        "rel_delta_MAE": (
                            float(np.mean(dMAE_list) / MAE_base)
                            if MAE_base != 0
                            else np.nan
                        ),
                    }
                )

        # Grouped PI on the same test window
        df_pi_groups = permutation_importance_by_group(
            fitted_estimator=model,
            X_test=Xt_te,
            y_test=y[te_idx],
            groups=CONFIG["feature_groups"],
            n_repeats=CONFIG["pi"]["n_repeats"],
            block_len=CONFIG["pi"]["block_len"],
            random_state=CONFIG["random_state"],
        )
        df_pi_groups["fold"] = fold_id
        pi_group_rows.extend(df_pi_groups.to_dict(orient="records"))

    fold_metrics_df = pd.DataFrame(fold_rows)

    pi_folds_df = pd.DataFrame(pi_rows) if compute_pi else None
    pi_summary_df = (
        (
            pi_folds_df.groupby("feature")[
                ["rel_delta_MAE", "rel_delta_R2", "delta_R2", "delta_MAE"]
            ]
            .median()
            .sort_values("rel_delta_MAE", ascending=False)
            .reset_index()
        )
        if compute_pi and not pi_folds_df.empty
        else None
    )

    pi_groups_folds_df = pd.DataFrame(pi_group_rows) if compute_pi else None
    pi_groups_summary_df = (
        (
            pi_groups_folds_df.groupby("group")[
                ["rel_delta_MAE", "rel_delta_R2", "delta_R2", "delta_MAE"]
            ]
            .median()
            .sort_values("rel_delta_MAE", ascending=False)
            .reset_index()
        )
        if compute_pi
        and (pi_groups_folds_df is not None)
        and (not pi_groups_folds_df.empty)
        else None
    )

    return {
        "fold_metrics_df": fold_metrics_df,
        "pi_folds_df": pi_folds_df,
        "pi_summary_df": pi_summary_df,
        "pi_groups_folds_df": pi_groups_folds_df,
        "pi_groups_summary_df": pi_groups_summary_df,
    }


# Save-Helper für die Ergebnisse aus run_explain_over_splits
def save_explain_outputs(
    res_dict: dict, results_dir: str, pseudonym: str, model_tag: str
):
    os.makedirs(results_dir, exist_ok=True)

    # Folds-Metriken
    fm = res_dict.get("fold_metrics_df", None)
    if fm is not None and not fm.empty:
        fm.to_csv(
            os.path.join(results_dir, f"{pseudonym}_{model_tag}_fold_metrics.csv"),
            index=False,
        )

    # Permutation Importance
    pf = res_dict.get("pi_folds_df", None)
    ps = res_dict.get("pi_summary_df", None)
    if pf is not None and not pf.empty:
        pf.to_csv(
            os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_feat_folds.csv"),
            index=False,
        )
    if ps is not None and not ps.empty:
        ps.to_csv(
            os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_feat_summary.csv"),
            index=False,
        )

    # Grouped Permutation Importance
    pgf = res_dict.get("pi_groups_folds_df", None)
    pgs = res_dict.get("pi_groups_summary_df", None)
    if pgf is not None and not pgf.empty:
        pgf.to_csv(
            os.path.join(results_dir, f"{pseudonym}_{model_tag}_perm_group_folds.csv"),
            index=False,
        )
    if pgs is not None and not pgs.empty:
        pgs.to_csv(
            os.path.join(
                results_dir, f"{pseudonym}_{model_tag}_perm_group_summary.csv"
            ),
            index=False,
        )


def days_since_start(group):
    start_date = group["timestamp_utc"].min()
    return (group["timestamp_utc"] - start_date).dt.days


def _time_cutoff_by_fraction(ts: pd.Series, frac: float) -> pd.Timestamp:
    if ts.empty:
        raise ValueError("Empty timestamp series.")
    t0, t1 = ts.min(), ts.max()
    if pd.isna(t0) or pd.isna(t1) or t0 == t1:
        return t1
    return t0 + (t1 - t0) * float(frac)


def rolling_time_splits_by_fraction(
    ts: pd.Series,
    init_frac_for_hp: float,  # e.g., CONFIG["init_frac_for_hp"]
    test_days: int = 30,  # fixed test length
    step_days: int = 30,  # extend training end by one month per fold
    embargo_days: int = 0,
    min_test_samples: int = 1,
):
    """
    Expanding, time-based rolling-origin splits:

      - Initial train window = fraction of total time span (init_frac_for_hp)
      - Each fold extends train_end by exactly `step_days` (30) days
      - Test window is always `test_days` (30) days
    """
    assert pd.api.types.is_datetime64_any_dtype(ts), "ts must be datetime-like"
    if ts.empty:
        return []

    t_min, t_max = ts.min(), ts.max()
    train_from = t_min
    # initial training end by fraction of time span:
    train_to = _time_cutoff_by_fraction(ts, init_frac_for_hp)

    td_test = timedelta(days=int(test_days))
    td_step = timedelta(days=int(step_days))
    td_emb = timedelta(days=int(embargo_days))

    splits = []
    while train_from < t_max:
        test_from = train_to + td_emb
        if test_from > t_max:
            break
        test_to = min(test_from + td_test, t_max)

        tr_idx = ts.index[(ts >= train_from) & (ts < train_to)].to_numpy()
        te_idx = ts.index[(ts >= test_from) & (ts < test_to)].to_numpy()

        if te_idx.size >= min_test_samples:
            splits.append(
                (
                    tr_idx,
                    te_idx,
                    {
                        "train_from_ts": pd.Timestamp(train_from),
                        "train_to_ts": pd.Timestamp(train_to),
                        "test_from_ts": pd.Timestamp(test_from),
                        "test_to_ts": pd.Timestamp(test_to),
                    },
                )
            )

        if test_to >= t_max:
            break

        # extend training end by one month for the next fold
        train_to = min(train_to + td_step, t_max)

    return splits


def _clean_pair(a, b, weights=None):
    """Align arrays, drop NaNs, and return (a, b, w) as np arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if weights is None:
        w = np.ones_like(a, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != a.shape:
            raise ValueError("weights must have same shape as metrics arrays")

    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w >= 0)
    return a[mask], b[mask], w[mask]


def _weighted_mean(x, w):
    wsum = np.sum(w)
    if wsum <= 0:
        return np.nan
    return float(np.sum(w * x) / wsum)


def bootstrap_paired_mean_diff(
    diffs,
    weights=None,
    B=10000,
    ci=(2.5, 97.5),
    alternative="greater",  # 'greater' means H1: mean(diffs) > 0
    seed=42,
):
    """
    Bootstrap the mean of paired differences (model - baseline).
      - For R²: diffs = R2_model - R2_dummy
      - For MAE: diffs = MAE_dummy - MAE_model  (so 'greater' still tests improvement)

    Returns:
      dict with mean_diff, ci_low, ci_high, p_value (one-sided), n
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)

    if weights is None:
        weights = np.ones_like(diffs, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != diffs.shape:
            raise ValueError("weights must match diffs shape")
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and non-negative")

    # Keep finite observations only
    m = np.isfinite(diffs) & np.isfinite(weights)
    diffs, weights = diffs[m], weights[m]
    n = diffs.size
    if n < 2:
        return {
            "mean_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n": int(n),
        }

    # Point estimate (weighted mean)
    mean_hat = _weighted_mean(diffs, weights)

    # Bootstrap resamples of the weighted mean
    idx = np.arange(n)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        # sample pairs with replacement
        sample_idx = rng.choice(idx, size=n, replace=True)
        boot[b] = _weighted_mean(diffs[sample_idx], weights[sample_idx])

    # Two-sided CI from bootstrap percentiles
    ci_low, ci_high = np.percentile(boot, ci)

    # One-sided p-value for 'greater' or 'less'
    if alternative == "greater":
        # proportion of bootstrap means <= 0
        p = float(np.mean(boot <= 0.0))
    elif alternative == "less":
        # proportion of bootstrap means >= 0
        p = float(np.mean(boot >= 0.0))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    return {
        "mean_diff": float(mean_hat),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p,
        "n": int(n),
    }


def compare_model_to_dummy_with_bootstrap(
    r2_model, r2_dummy, mae_model, mae_dummy, n_test_samples=None, B=10000, seed=42
):
    """
    Convenience wrapper:
      - R² improvement:  diff_R2  = R2_model - R2_dummy   (want > 0)
      - MAE improvement: diff_MAE = MAE_dummy - MAE_model (want > 0)
    If provided, n_test_samples are used as fold weights (recommended for unequal test sizes).
    """
    # Clean and align inputs
    r2_m, r2_d, w_r2 = _clean_pair(r2_model, r2_dummy, n_test_samples)
    mae_m, mae_d, w_mae = _clean_pair(mae_model, mae_dummy, n_test_samples)

    # Build differences with "improvement is positive"
    diff_r2 = r2_m - r2_d
    diff_mae = mae_d - mae_m

    res_r2 = bootstrap_paired_mean_diff(
        diffs=diff_r2, weights=w_r2, B=B, alternative="greater", seed=seed
    )
    res_mae = bootstrap_paired_mean_diff(
        diffs=diff_mae, weights=w_mae, B=B, alternative="greater", seed=seed
    )

    return {
        "R2": res_r2,
        "MAE": res_mae,
    }


def test_elasticnet_outperforms_dummy_regressor(X, y, pseudonym):
    X_preprocessed = preprocess_pipeline().fit_transform(X)
    rtscv = RepeatedTimeSeriesSplit(
        n_splits=10,
        n_repeats=5,
        max_offset_frac=0.2,
        gap=0,
        random_state=RANDOM_STATE,
        offset_strategy="uniform",   # oder "linspace"
        test_size=30,                # NEU: fixe Testfenstergröße (optional)
        min_train_size=60,           # NEU: sinnvolle Untergrenze (z.B. 2 Monate)
        min_test_size=14,            # NEU: mind. 2 Wochen
        warn=True,
        return_metadata=False,       # auf True setzen, wenn du die Meta brauchst
    )
    print(f"[RepeatedTSS] nominal={rtscv.get_n_splits()} "
        f"actual={rtscv.actual_n_splits(len(X_preprocessed))}")


    en = ElasticNet(
        max_iter=CONFIG["en"]["max_iter"],
        tol=CONFIG["en"]["tol"],
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "alpha": CONFIG["en"]["param_grid"]["model__alpha"],
        "l1_ratio": CONFIG["en"]["param_grid"]["model__l1_ratio"],
    }

    en_gs = GridSearchCV(
        estimator=en,
        param_grid=param_grid,
        cv=rtscv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        scoring="r2",
    )

    en_gs.fit(X_preprocessed, y)

    # ---- paste or import the bootstrap functions from earlier ----
    # (bootstrap_paired_mean_diff, compare_model_to_dummy_with_bootstrap)
    # Assume they are available in scope. If not, paste them above.

    # 1) Extract best hyperparameters from GridSearchCV
    best_alpha = en_gs.best_params_.get("alpha", en_gs.best_params_.get("model__alpha"))
    best_l1_ratio = en_gs.best_params_.get(
        "l1_ratio", en_gs.best_params_.get("model__l1_ratio")
    )

    en_best = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        max_iter=CONFIG["en"]["max_iter"],
        tol=CONFIG["en"]["tol"],
        random_state=RANDOM_STATE,
    )

    # 2) Manual CV over the SAME tscv to get fold-wise metrics  (pandas-safe)
    r2_model_folds, r2_dummy_folds = [], []
    mae_model_folds, mae_dummy_folds = [], []
    n_test_folds = []

    for fold_id, (tr_idx, te_idx) in enumerate(rtscv.split(X_preprocessed), start=1):
        # IMPORTANT: use .iloc for positional indexing on DataFrames
        X_tr = X_preprocessed.iloc[tr_idx, :]
        X_te = X_preprocessed.iloc[te_idx, :]
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        # Fit Elastic Net with best params on the fold's TRAIN
        en_best.fit(X_tr, y_tr)
        y_pred_model = en_best.predict(X_te)

        # Baseline: DummyRegressor (mean of TRAIN y)
        dum = DummyRegressor(strategy="mean")
        dum.fit(
            y_tr.reshape(-1, 1) if y_tr.ndim == 1 else y_tr, y_tr
        )  # sklearn ignores X for Dummy
        # Simpler/clearer:
        dum = DummyRegressor(strategy="mean").fit(np.zeros((len(y_tr), 1)), y_tr)
        y_pred_dummy = dum.predict(np.zeros((len(y_te), 1)))

        # Metrics
        r2_model_folds.append(r2_score(y_te, y_pred_model))
        r2_dummy_folds.append(r2_score(y_te, y_pred_dummy))
        mae_model_folds.append(mean_absolute_error(y_te, y_pred_model))
        mae_dummy_folds.append(mean_absolute_error(y_te, y_pred_dummy))
        n_test_folds.append(len(te_idx))

    r2_model_folds = np.asarray(r2_model_folds, dtype=float)
    r2_dummy_folds = np.asarray(r2_dummy_folds, dtype=float)
    mae_model_folds = np.asarray(mae_model_folds, dtype=float)
    mae_dummy_folds = np.asarray(mae_dummy_folds, dtype=float)
    n_test_folds = np.asarray(n_test_folds, dtype=int)

    # 3) Bootstrap: is EN significantly better than Dummy?
    boot_results = compare_model_to_dummy_with_bootstrap(
        r2_model=r2_model_folds,
        r2_dummy=r2_dummy_folds,
        mae_model=mae_model_folds,
        mae_dummy=mae_dummy_folds,
        n_test_samples=n_test_folds,  # weights per fold (recommended)
        B=20000,
        seed=123,
    )

    # 4) Print a concise report
    r2_res = boot_results["R2"]
    mae_res = boot_results["MAE"]

    print("=== Elastic Net vs Dummy (paired bootstrap over CV folds) ===")
    print(f"Best params: alpha={best_alpha}, l1_ratio={best_l1_ratio}")
    print(
        f"ΔR²  (EN − Dummy): mean={r2_res['mean_diff']:.3f}, "
        f"95% CI=[{r2_res['ci_low']:.3f}, {r2_res['ci_high']:.3f}], "
        f"one-sided p={r2_res['p_value']:.4f}, n_folds={r2_res['n']}"
    )
    print(
        f"ΔMAE (Dummy − EN): mean={mae_res['mean_diff']:.3f}, "
        f"95% CI=[{mae_res['ci_low']:.3f}, {mae_res['ci_high']:.3f}], "
        f"one-sided p={mae_res['p_value']:.4f}, n_folds={mae_res['n']}"
    )

    # Optional: make a tidy DataFrame of per-fold metrics for logging
    per_fold_df = pd.DataFrame(
        {
            "fold": np.arange(1, len(r2_model_folds) + 1),
            "r2_en": r2_model_folds,
            "r2_dummy": r2_dummy_folds,
            "mae_en": mae_model_folds,
            "mae_dummy": mae_dummy_folds,
            "n_test": n_test_folds,
            "delta_r2": r2_model_folds - r2_dummy_folds,
            "delta_mae": mae_dummy_folds - mae_model_folds,  # improvement if > 0
        }
    )
    print(per_fold_df)

    dummy_r2 = []
    model_r2 = []

    for train_idx, test_idx in rtscv.split(X_preprocessed):
        X_train, X_test = X_preprocessed.iloc[train_idx], X_preprocessed.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Best EN from CV
        best_en = ElasticNet(
            **en_gs.best_params_, max_iter=10000, tol=1e-3, random_state=42
        )
        best_en.fit(X_train, y_train)
        y_pred_en = best_en.predict(X_test)
        model_r2.append(r2_score(y_test, y_pred_en))

        # Dummy baseline
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        dummy_r2.append(r2_score(y_test, y_pred_dummy))

    plot_elasticnet_vs_dummyregressor(
        per_fold_df=per_fold_df,
        boot_results=boot_results,
        out_path=f"{RESULTS_DIR}/{pseudonym}_elasticnet_vs_dummy.png",
    )

    # significance thresholds
    ALPHA = 0.05  # one-sided test level

    r2_res = boot_results["R2"]
    mae_res = boot_results["MAE"]

    def _safe_pos(x):
        try:
            return (x is not None) and np.isfinite(x) and (x > 0)
        except Exception:
            return False

    # Require BOTH: one-sided p-value < alpha AND 95% CI excludes 0 on the good side
    is_significant_r2 = (
        (r2_res["p_value"] < ALPHA)
        and _safe_pos(r2_res["mean_diff"])
        and _safe_pos(r2_res["ci_low"])
    )

    is_significant_mae = (
        (mae_res["p_value"] < ALPHA)
        and _safe_pos(mae_res["mean_diff"])
        and _safe_pos(mae_res["ci_low"])
    )

    # Optional: if you want Bonferroni for the two metrics, uncomment:
    # ALPHA_EACH = ALPHA / 2.0
    # is_significant_r2 = (r2_res["p_value"] < ALPHA_EACH) and _safe_pos(r2_res["ci_low"])
    # is_significant_mae = (mae_res["p_value"] < ALPHA_EACH) and _safe_pos(mae_res["ci_low"])

    proceed = is_significant_r2 or is_significant_mae

    if proceed:
        print(
            "✅ Elastic Net performs significantly better than DummyRegressor "
            f"({'R²' if is_significant_r2 else ''}{' & ' if is_significant_r2 and is_significant_mae else ''}"
            f"{'MAE' if is_significant_mae else ''})."
        )
        run_further_analysis = True
    else:
        print("⚠️ Elastic Net not significantly better — skipping downstream steps.")
        run_further_analysis = False

    return run_further_analysis


# %% Main Processing Function ===================================================
def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    plot_data = {}

    ensure_results_dir()

    for pseudonym in pseudonyms:
        print("========================================")
        print(f"Processing {pseudonym}")
        print("========================================")
        df_participant = df_raw[df_raw["pseudonym"] == pseudonym].copy()
        print(f"[INFO] {pseudonym}: n_days={df_participant.shape[0]}")
        rmssd_phq2 = compute_rmssd(df_participant[PHQ2_COLUMN].values)
        rmssd_values_phq2.append(rmssd_phq2)

        # --- Daten vorbereiten (nur gültige Zielwerte) ---
        X, y, _ = prepare_data(df_participant, target_column)
        mask_valid = np.isfinite(y)
        X = X.loc[mask_valid].reset_index(drop=True)
        y = np.asarray(y)[mask_valid]
        ts_valid = df_participant.loc[mask_valid, "timestamp_utc"].reset_index(
            drop=True
        )

        elasticnet_outperforms_dummy_regressor = (
            test_elasticnet_outperforms_dummy_regressor(X, y, pseudonym)
        )

        if not elasticnet_outperforms_dummy_regressor:
            print(
                f"[INFO] {pseudonym}: ElasticNet does not outperform Dummy; skipping."
            )
            continue

        n = len(y)
        if n < 30:
            print(f"[WARN] {pseudonym}: zu wenige Datenpunkte ({n}), überspringe.")
            continue

        # --- time-based initial split by CONFIG["init_frac_for_hp"] ----------
        train_frac = CONFIG["init_frac_for_hp"]  # e.g., 0.70 of time span
        cut_ts = _time_cutoff_by_fraction(ts_valid, train_frac)

        idx_train = np.where(ts_valid <= cut_ts)[0]
        idx_test = np.where(ts_valid > cut_ts)[0]

        if idx_test.size == 0:
            print(f"[WARN] {pseudonym}: no holdout after time cutoff; skipping.")
            continue

        X_train_df = X.iloc[idx_train].copy()
        X_test_df = X.iloc[idx_test].copy()
        y_train = y[idx_train]
        y_test = y[idx_test]

        print(
            f"[INFO] time cutoff @ {cut_ts} "
            f"→ train: {idx_train.size} samples, test: {idx_test.size} samples"
        )

        # Train-only Preprocessing (fit on initial train window)
        pp = preprocess_pipeline()
        X_init_for_hp = pp.fit_transform(X_train_df)

        print(f"{train_frac}:var(y_train) =", float(np.var(y_train)))
        print(f"{1-train_frac}:var(y_test) =", float(np.var(y_test)))
        dum = DummyRegressor(strategy="mean").fit(X_train_df, y_train)
        print("baseline R2 =", r2_score(y_test, dum.predict(X_test_df)))

        # Train-only Preprocessing (Imputation) auf Startfenster
        pp = preprocess_pipeline()
        X_init_for_hp = pp.fit_transform(X_train_df)

        en = ElasticNet(
            max_iter=CONFIG["en"]["max_iter"],
            tol=CONFIG["en"]["tol"],
            random_state=RANDOM_STATE,
        )

        param_grid = {
            "alpha": CONFIG["en"]["param_grid"]["model__alpha"],
            "l1_ratio": CONFIG["en"]["param_grid"]["model__l1_ratio"],
        }

        en_gs = GridSearchCV(
            estimator=en,
            param_grid=param_grid,
            cv=TimeSeriesSplit(
                n_splits=CONFIG["cv_n_splits"], gap=CONFIG["cv_embargo"]
            ),
            n_jobs=-1,
            refit=True,
            verbose=0,
            scoring="r2",
        )

        en_gs.fit(X_init_for_hp, y_train)
        best_params = en_gs.best_params_
        best_alpha = float(best_params.get("alpha", best_params.get("model__alpha")))
        best_l1 = float(best_params.get("l1_ratio", best_params.get("model__l1_ratio")))

        ALPHA_CAP = 10.0
        if best_alpha > ALPHA_CAP:
            print(f"[Init] alpha {best_alpha:.4g} capped → {ALPHA_CAP}")
            best_alpha = ALPHA_CAP

        print(f"[Init-HP] EN alpha={best_alpha:.4g}, l1_ratio={best_l1}")

        # =================== ELASTIC NET: (fixe HP aus Startfenster) ====================
        print("[EN] Fit mit fixen HP (aus Startfenster) ...")

        # --- PREPROCESSING: train-only pro Split fitten, danach optional Fix-Scaler anwenden
        Xt_train_en = X_init_for_hp
        Xt_test_en = pp.transform(X_test_df)

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

        # === DEBUG: ElasticNet Diagnose (direkt nach r2_elastic / mae_elastic) ===
        def _is_near_const(arr, tol_var=1e-12):
            return float(np.var(arr)) <= tol_var

        nz = int(np.count_nonzero(model_en.coef_))
        interc = float(model_en.intercept_)
        pred_var = float(np.var(y_pred_en))
        ytr_var = float(np.var(y_train))
        yte_var = float(np.var(y_test))

        print(
            "[EN][DEBUG]",
            f"nz-coef={nz}",
            f"intercept={interc:.4g}",
            f"best_alpha={best_alpha:.4g}",
            f"best_l1={best_l1}",
            f"var(y_train)={ytr_var:.4g}",
            f"var(y_test)={yte_var:.4g}",
            f"var(y_pred)={pred_var:.4g}",
            f"R2={r2_elastic:.3f}",
            f"MAE={mae_elastic:.3f}",
        )

        # Baseline zum Vergleich
        y_pred_mean = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
        r2_base = r2_score(y_test, y_pred_mean)
        mae_base = mean_absolute_error(y_test, y_pred_mean)
        print(
            "[EN][DEBUG] baseline(mean-of-train): R2={r2:.3f}, MAE={mae:.3f}".format(
                r2=r2_base, mae=mae_base
            )
        )

        # Checks & Hinweise
        if nz == 0 and abs(interc) < 1e-8:
            print(
                "[EN][WARN] Nullmodell: alle Koeffizienten = 0 und Intercept ≈ 0 → konstante Vorhersage 0."
            )
            print(
                "           Ursachen: sehr großes alpha oder sehr geringe Varianz im Startfenster/Train."
            )

        if _is_near_const(y_pred_en):
            print("[EN][WARN] Vorhersagen sind (nahezu) konstant. var(y_pred)≈0.")
            print(
                "           Prüfe alpha-Raster (zu groß?), l1_ratio (zu lasso-lastig?),"
            )
            print(
                "           bzw. Zielvarianz im Startfenster (ggf. Startfenster erweitern)."
            )

        if r2_elastic < r2_base:
            print(
                "[EN][INFO] Modell schlägt Baseline (Train-Mean) NICHT. Evtl. Underfitting/zu starke Regularisierung."
            )
            print(
                "           Tipp: feineres alpha (z.B. 1e-6..1e2), l1_ratio etwas ridge-iger (<=0.3),"
            )
            print("           oder Startfenster-Varianz prüfen/erweitern.")

        # Extremwerte/Skalen sanity checks
        print(
            "[EN][DEBUG] y_train[min,max,mean]={:.3f},{:.3f},{:.3f} | y_test[min,max,mean]={:.3f},{:.3f},{:.3f} | y_pred[min,max,mean]={:.3f},{:.3f},{:.3f}".format(
                float(np.min(y_train)),
                float(np.max(y_train)),
                float(np.mean(y_train)),
                float(np.min(y_test)),
                float(np.max(y_test)),
                float(np.mean(y_test)),
                float(np.min(y_pred_en)),
                float(np.max(y_pred_en)),
                float(np.mean(y_pred_en)),
            )
        )

        # Optional: kleinste/niedrigste Standardabweichungen in den train-Features (nach Preprocessing)
        try:
            # Falls Xt_train_en ein numpy-Array ist
            if not hasattr(Xt_train_en, "std"):
                Xtr_std = np.std(Xt_train_en, axis=0)
            else:
                Xtr_std = Xt_train_en.std(axis=0)
            min_std = float(np.min(Xtr_std))
            n_zero_std = int(np.sum(np.isclose(Xtr_std, 0)))
            print(
                f"[EN][DEBUG] Xt_train_en: min(std)={min_std:.3g}, #zero-std-cols={n_zero_std}"
            )
        except Exception as e:
            print("[EN][DEBUG] Feature-Std-Check skipped:", repr(e))

        print(
            f"EN R2={r2_elastic}, MAE={mae_elastic}, alpha={best_alpha}, l1_ratio={best_l1}"
        )
        print("EN # nonzero betas:", np.count_nonzero(model_en.coef_))

        # Modelle speichern
        joblib.dump(
            {"pre": pp, "model": model_en},
            os.path.join(RESULTS_DIR, "models", f"{pseudonym}_elasticnet.joblib"),
        )

        # ================= RANDOM FOREST: GridSearch NUR im Startfenster (+ fix) ====================
        print("[RF] GridSearchCV (TimeSeriesSplit) NUR Startfenster ...")

        rf_search_est = Pipeline(
            [("model", RandomForestRegressor(random_state=RANDOM_STATE))]
        )
        rf_gs = GridSearchCV(
            estimator=rf_search_est,
            param_grid=CONFIG["rf"]["param_grid"],
            scoring=CONFIG["rf"]["scoring"],
            cv=TimeSeriesSplit(n_splits=CONFIG["cv_n_splits"]),
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        rf_gs.fit(X_init_for_hp, y_train)
        rf_best_params = rf_gs.best_params_
        print("[Init-HP] RF best:", rf_best_params)

        rf_best_params_plain = {
            k.split("__", 1)[1]: v
            for k, v in rf_best_params.items()
            if k.startswith("model__")
        }

        Xt_train_rf = X_init_for_hp
        Xt_test_rf = pp.transform(X_test_df)

        model_rf = RandomForestRegressor(
            random_state=RANDOM_STATE, **rf_best_params_plain
        )
        model_rf.fit(Xt_train_rf, y_train)

        y_pred_rf = model_rf.predict(Xt_test_rf)
        r2_rf = round(r2_score(y_test, y_pred_rf), 2)
        mae_rf = round(mean_absolute_error(y_test, y_pred_rf), 1)

        print(
            f"RF R2={r2_rf}, MAE={mae_rf}, n_estimators={model_rf.n_estimators}, max_depth={model_rf.max_depth}, min_samples_leaf={model_rf.min_samples_leaf}"
        )
        # Modelle speichern
        joblib.dump(
            {"pre": pp, "model": model_rf},
            os.path.join(RESULTS_DIR, "models", f"{pseudonym}_rf.joblib"),
        )

        # ==================== Parity Plots & Zeitreihen-Plots ==================
        evaluate_and_plot_parity(
            RESULTS_DIR,
            y_test,
            y_pred_en,
            r2_elastic,
            mae_elastic,
            pseudonym,
            "elasticnet",
            "01",
        )
        evaluate_and_plot_parity(
            RESULTS_DIR, y_test, y_pred_rf, r2_rf, mae_rf, pseudonym, "rf", "03"
        )

        # Für Zeitreihen-Plot Roh/Pred (optional: nur Holdout anzeigen)
        timestamps_model = (
            df_participant.loc[mask_valid, "timestamp_utc"].astype(str).tolist()
        )
        individual_adherence = (
            df_participant.loc[
                mask_valid,
                [
                    "number_used_ibi_datapoints_rq1_day",
                    "number_used_ibi_datapoints_rq1_night",
                ],
            ].sum(axis=1)
            / df_participant.loc[
                mask_valid,
                [
                    "number_used_ibi_datapoints_rq1_day",
                    "number_used_ibi_datapoints_rq1_night",
                ],
            ]
            .sum(axis=1)
            .max()
        )
        global_adherence = (
            df_participant.loc[
                mask_valid,
                [
                    "number_used_ibi_datapoints_rq1_day",
                    "number_used_ibi_datapoints_rq1_night",
                ],
            ].sum(axis=1)
            / df_raw[
                [
                    "number_used_ibi_datapoints_rq1_day",
                    "number_used_ibi_datapoints_rq1_night",
                ]
            ]
            .sum(axis=1)
            .max()
        )
        plot_data[pseudonym] = {
            "timestamps": timestamps_model,
            "phq2_raw": y.tolist(),
            "individual_adherence": individual_adherence.tolist(),
            "global_adherence": global_adherence.tolist(),
            "train_mask": (ts_valid <= cut_ts).tolist(),
            "test_mask": (ts_valid > cut_ts).tolist(),
            "elastic": {
                "pred": list(model_en.predict(pp.transform(X))),
                "lower": None,
                "upper": None,
            },
            "rf": {
                "pred": list(model_rf.predict(pp.transform(X))),
                "lower": None,
                "upper": None,
            },
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

        # ==================== Rolling PI ========================
        print("  [EN] Rolling PI ...")

        make_en = lambda: ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1,
            max_iter=CONFIG["en"]["max_iter"],
            tol=CONFIG["en"]["tol"],
            random_state=RANDOM_STATE,
        )

        res_en = run_explain_over_splits(
            X_df=X,
            y=y,
            ts=ts_valid,
            make_model=make_en,
            compute_pi=True,
            hp_train_frac=CONFIG["init_frac_for_hp"],
            test_days=30,
            step_days=30,
            embargo_days=CONFIG["pi"]["embargo"],
            n_repeats=CONFIG["pi"]["n_repeats"],
            random_state=CONFIG["random_state"],
            fixed_preprocess_pipeline=pp,
        )

        print("  [RF] Rolling PI ...")

        def make_rf():
            return RandomForestRegressor(
                random_state=RANDOM_STATE,
                **{
                    k: rf_best_params[k]
                    for k in [
                        "n_estimators",
                        "max_depth",
                        "min_samples_leaf",
                        "max_features",
                    ]
                    if k in rf_best_params
                },
            )

        res_rf = run_explain_over_splits(
            X_df=X,
            y=y,
            ts=ts_valid,
            make_model=make_rf,
            compute_pi=True,
            hp_train_frac=CONFIG["init_frac_for_hp"],
            test_days=30,
            step_days=30,
            embargo_days=CONFIG["pi"]["embargo"],
            n_repeats=CONFIG["pi"]["n_repeats"],
            random_state=CONFIG["random_state"],
            fixed_preprocess_pipeline=pp,
        )

        # ============== Ergebnisse speichern (EN/RF) ==============
        save_explain_outputs(res_en, RESULTS_DIR, pseudonym, "EN")
        save_explain_outputs(res_rf, RESULTS_DIR, pseudonym, "RF")

        # ============== Zusammenfassungs-Plots (Top-K) ==============
        # EN – PI Bars
        if (
            res_en.get("pi_summary_df") is not None
            and not res_en["pi_summary_df"].empty
        ):
            plot_importance_bars(
                res_en["pi_summary_df"],
                "rel_delta_R2",
                title=f"{pseudonym} EN – Permutation Importance ΔR² (median über Folds)",
                outpath=os.path.join(RESULTS_DIR, f"{pseudonym}_EN_pi_summary_bar.png"),
                top_k=TOP_K,
            )

        # RF – PI Bars
        if (
            res_rf.get("pi_summary_df") is not None
            and not res_rf["pi_summary_df"].empty
        ):
            plot_importance_bars(
                res_rf["pi_summary_df"],
                "rel_delta_R2",
                title=f"{pseudonym} RF – Permutation Importance ΔR² (median über Folds)",
                outpath=os.path.join(RESULTS_DIR, f"{pseudonym}_RF_pi_summary_bar.png"),
                top_k=TOP_K,
            )

        # ============== Folds auf PHQ-2 Zeitreihe ==============
        # timestamps passend zu X_df_valid / y_valid:
        timestamps_valid = df_participant.loc[
            df_participant[target_column].notna(), "timestamp_utc"
        ].reset_index(drop=True)

        # EN-Folds
        if (
            res_en.get("fold_metrics_df") is not None
            and not res_en["fold_metrics_df"].empty
        ):
            plot_folds_on_timeseries(
                timestamps=timestamps_valid,
                y_values=y,
                fold_metrics_df=res_en["fold_metrics_df"],
                out_path=os.path.join(
                    RESULTS_DIR, f"{pseudonym}_EN_folds_timeseries.png"
                ),
                title=f"{pseudonym} – EN: Test-Fenster & Metriken",
            )

        # RF-Folds
        if (
            res_rf.get("fold_metrics_df") is not None
            and not res_rf["fold_metrics_df"].empty
        ):
            plot_folds_on_timeseries(
                timestamps=timestamps_valid,
                y_values=y,
                fold_metrics_df=res_rf["fold_metrics_df"],
                out_path=os.path.join(
                    RESULTS_DIR, f"{pseudonym}_RF_folds_timeseries.png"
                ),
                title=f"{pseudonym} – RF: Test-Fenster & Metriken",
            )

    return results, plot_data


if __name__ == "__main__":
    # %% Run the pipeline ===========================================================
    # Load dataset
    df_raw = pd.read_pickle(DATA_PATH)
    df_raw["day_of_week"] = df_raw["timestamp_utc"].dt.weekday
    df_raw["month_of_year"] = df_raw["timestamp_utc"].dt.month
    df_raw["day_since_start"] = df_raw.groupby("patient_id", group_keys=False).apply(
        days_since_start
    )

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
    if os.path.exists(CACHE_RESULTS_PATH) and LOAD_CACHED_RESULTS:
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
    plot_phq2_timeseries_with_adherence_from_results(RESULTS_DIR, plot_data, "elastic")
    plot_phq2_test_errors_from_results(
        RESULTS_DIR, plot_data, "elastic", show_pred_ci=False
    )
    plot_phq2_test_errors_from_results(
        RESULTS_DIR, plot_data, "elastic", show_pred_ci=False
    )

    plot_phq2_timeseries_from_results(RESULTS_DIR, plot_data, "rf")
    plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf", show_pred_ci=False)
    plot_phq2_test_errors_from_results(RESULTS_DIR, plot_data, "rf", show_pred_ci=False)

    # %% Save evaluation metrics (per participant) ==================================
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.to_csv(f"{RESULTS_DIR}/model_performance_summary.csv")

    # %%
