# %% Data Loading and Setup =====================================================
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
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

set_config(transform_output="pandas")

# ==== RUN/PROJECT CONFIG ======================================================
CONFIG = {
    "random_state": 42,
    "top_k": 15,
    "paths": {
        "data": "data/df_merged_dupsfree_v8.pickle",
        "results_dir": "results_with_nightfeatures_perfeaturescaler_timeaware_debug2",
    },
    "targets": {
        "phq2": "abend_PHQ2_sum",
        "phq9": "woche_PHQ9_sum",
    },
    # Anteil Startfenster für HP-Tuning und Fix-Scaler-Fit
    "init_frac_for_hp": 0.70,
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

    Notes
    -----
    - Der globale, fixe Scaler (falls genutzt) wird separat bereitgestellt.
    - So bleiben β mit fixem Scaler über Zeit vergleichbar, während
      Imputation & per-Feature-Skalierung weiterhin train-only bleiben.
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


def ensure_var_in_init(y, init_end, min_var=1e-3, step_days=14, max_frac=0.5):
    """
    Erweitert das Startfenster iterativ, bis Var(y) >= min_var oder max_frac erreicht.

    Wenn die Zielvarianz im Startfenster zu klein ist (Nullmodell-Risiko),
    wird init_end in Schritten (step_days) erhöht, bis mindestens min_var
    erreicht ist oder höchstens max_frac * len(y) genutzt wurden.

    Parameters
    ----------
    y : array-like
        Zielreihe (nach Maskierung, Länge = n).
    init_end : int
        Aktuelles Ende des Startfensters (exklusiv).
    min_var : float
        Minimale Varianzschwelle.
    step_days : int
        Schrittweite zur Erweiterung (Tage).
    max_frac : float
        Obergrenze als Anteil der Gesamtlänge.

    Returns
    -------
    int
        Neues init_end.
    """
    import numpy as np

    n = len(y)
    max_end = int(max_frac * n)
    while np.var(y[:init_end]) < min_var and (init_end + step_days) <= max_end:
        init_end += step_days
    return init_end


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


def run_feature_permutation_importance_over_splits(
    X_df: pd.DataFrame,
    y: np.ndarray,
    test_size: int,
    hp_train_frac: float,
    embargo: int,
    estimator: BaseEstimator,  # bereits mit finalen HP konfiguriert
    n_repeats: int = 50,
    agg: str = "mean",
    random_state: int = 42,
    fixed_preprocess_pipeline: Pipeline | None = None,
):
    n = len(y)
    splits = rolling_origin_splits(
        n_samples=n,
        test_size=test_size,
        hp_train_frac=hp_train_frac,
        embargo=embargo,
        step_train=30,
        min_test_size=5,
    )

    all_rows = []
    for fold_id, (tr, te) in enumerate(splits, start=1):
        Xt_tr = fixed_preprocess_pipeline.fit_transform(X_df.iloc[tr])
        Xt_te = fixed_preprocess_pipeline.transform(X_df.iloc[te])

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
    make_model,  # callable -> unfit model (EN/RF/...)
    compute_pi: bool = True,
    test_size: int = 30,
    hp_train_frac: float = 0.3,
    embargo: int = 0,
    n_repeats: int = 50,
    random_state: int = 42,
    fixed_preprocess_pipeline: Pipeline | None = None,
):
    """
    Expanding-/Rolling-Evaluation über zeitlich geordnete Splits mit optionaler
    (blockierter) Permutation Importance auf dem Testfenster.

    Für eine chronologisch sortierte Tagesreihe erzeugt die Funktion Rolling-Origin-Splits
    (Expanding per Default), trainiert je Fold eine Preprocessing+Modell-Pipeline auf dem
    Trainingsfenster, evaluiert auf dem Testfenster (R²/MAE) und berechnet optional
    blockierte Permutation Importances. Zusätzlich kann eine
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
    compute_pi : bool, optional
        Ob pro Fold Permutation Importances berechnet werden (Feature-weise + Gruppen-weise).
    test_size : int, optional
        Länge des Testfensters je Fold (in Tagen).
    embargo : int, optional
        Sicherheitslücke zwischen Train-Ende und Test-Start (Tage), um Leakage zu vermeiden.
    n_repeats : int, optional
        Wiederholungen pro Feature/Gruppe für PI (Monte-Carlo-Schätzung).
    random_state : int, optional
        Seed für Reproduzierbarkeit (wird für Permutation verwendet).
    fixed_preprocess_pipeline : Pipeline | None, optional
        Optionale, bereits gefittete Preprocessing-Pipeline, die **in allen Folds
        unverändert** auf Train/Test angewandt wird (für konsistente Feature-Skalen
        und damit vergleichbare PI über Folds).

    Returns
    -------
    dict
        {
          "fold_metrics_df":  DataFrame mit R²/MAE je Fold,
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
    - PI werden für 100%-Darstellungen fold-normalisiert (Summe je Fold = 1):
        PI: rel_delta_R2.
        Die Summary-Tabellen basieren auf Medianen dieser relativen Werte, was robuste, vergleichbare Aggregation
        über Folds (und später über Personen) erlaubt.

    Example
    -------
    >>> CONFIG["pi"] = {"block_len": 7, "n_repeats": 50}
    >>> CONFIG["feature_groups"] = COARSE_GROUPS   # oder DAY_NIGHT_GROUPS
    >>> res = run_explain_over_splits(
    ...     X_df, y, make_model=make_en_pipeline, compute_pi=True,
    ...     n_splits=5, test_size=14, min_train_size=90, random_state=42
    ... )
    >>> res["pi_groups_summary_df"].head()
    """

    splits = rolling_origin_splits(
        n_samples=len(y),
        test_size=test_size,
        hp_train_frac=hp_train_frac,
        embargo=embargo,
        step_train=30,
        min_test_size=5,
    )

    rng = np.random.default_rng(random_state)
    fold_rows = []
    pi_rows = []
    pi_group_rows = []

    for fold_id, (tr, te) in enumerate(splits, start=1):
        Xt_tr = fixed_preprocess_pipeline.transform(X_df.iloc[tr])  # train-only fit
        Xt_te = fixed_preprocess_pipeline.transform(X_df.iloc[te])

        # --- Modell fitten
        model = make_model()
        model.fit(Xt_tr, y[tr])

        # --- Test-Metriken
        y_hat = model.predict(Xt_te)
        R2 = r2_score(y[te], y_hat)
        MAE = mean_absolute_error(y[te], y_hat)
        fold_rows.append(
            {
                "fold": fold_id,
                "r2": float(R2),
                "mae": float(MAE),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "train_end_idx": int(tr[-1]),
                "test_start_idx": int(te[0]),
                "test_end_idx": int(te[-1]),
            }
        )

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
                    Xp = _block_permute_col(
                        Xt_te, j, block_len=CONFIG["pi"]["block_len"], rng=rng
                    )
                    y_hat_p = model.predict(Xp)
                    dR2_list.append(R2_base - r2_score(y[te], y_hat_p))
                    dMAE_list.append(mean_absolute_error(y[te], y_hat_p) - MAE_base)
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
        # --- Gruppenweise Permutation Importance (blockweise)

        df_pi_groups = permutation_importance_by_group(
            fitted_estimator=model,
            X_test=Xt_te,
            y_test=y[te],
            groups=CONFIG["feature_groups"],  # TODO: into fun args?
            n_repeats=CONFIG["pi"]["n_repeats"],
            block_len=CONFIG["pi"]["block_len"],
            random_state=CONFIG["random_state"],
        )
        df_pi_groups["fold"] = fold_id
        pi_group_rows.extend(df_pi_groups.to_dict(orient="records"))

    # Ergebnisse zusammenfassen
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
        and pi_groups_folds_df is not None
        and not pi_groups_folds_df.empty
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


# %% Main Processing Function ===================================================
def process_participants(df_raw, pseudonyms, target_column):
    results = {}
    rmssd_values_phq2 = []
    plot_data = {}

    df_raw["day_of_week"] = df_raw["timestamp_utc"].dt.weekday
    df_raw["month_of_year"] = df_raw["timestamp_utc"].dt.month
    df_raw["day_since_start"] = df_raw.groupby("pseudonym", group_keys=False).apply(
        days_since_start
    )

    ensure_results_dir()

    for pseudonym in ["bubblypie2"]:
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
        n = len(y)
        if n < 30:
            print(f"[WARN] {pseudonym}: zu wenige Datenpunkte ({n}), überspringe.")
            continue

        # --- Dummy Regressor ---
        train_frac = CONFIG["init_frac_for_hp"]
        gap = 0
        split = int(round(train_frac * n))
        idx_train = np.arange(0, split - gap)
        idx_test = np.arange(split, n)

        X_train_df = X.iloc[idx_train].copy()
        X_test_df = X.iloc[idx_test].copy()
        y_train = y[idx_train]
        y_test = y[idx_test]

        print(f"{train_frac}:var(y_train) =", float(np.var(y_train)))
        print(f"{1-train_frac}:var(y_test) =", float(np.var(y_test)))
        dum = DummyRegressor(strategy="mean").fit(X_train_df, y_train)
        print("baseline R2 =", r2_score(y_test, dum.predict(X_test_df)))

        # Train-only Preprocessing (Imputation) auf Startfenster
        pp = preprocess_pipeline()
        X_init_for_cv = pp.fit_transform(X_train_df)

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

        en_gs.fit(X_init_for_cv, y_train)
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
        Xt_train_en = X_init_for_cv
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
        rf_gs.fit(X_init_for_cv, y_train)
        rf_best_params = rf_gs.best_params_
        print("[Init-HP] RF best:", rf_best_params)

        rf_best_params_plain = {
            k.split("__", 1)[1]: v
            for k, v in rf_best_params.items()
            if k.startswith("model__")
        }

        Xt_train_rf = X_init_for_cv
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
            "train_mask": [i < split for i in range(n)],
            "test_mask": [i >= split for i in range(n)],
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
            make_model=make_en,
            compute_pi=True,
            test_size=CONFIG["pi"]["test_size"],
            hp_train_frac=CONFIG["init_frac_for_hp"],
            embargo=CONFIG["pi"]["embargo"],
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
            make_model=make_rf,
            compute_pi=True,
            test_size=CONFIG["pi"]["test_size"],
            hp_train_frac=CONFIG["init_frac_for_hp"],
            embargo=CONFIG["pi"]["embargo"],
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
