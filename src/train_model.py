"""
Vessel delay prediction — model training and evaluation.

Trains two classifiers on the point-in-time correct feature table produced by
feature_engineering.py, evaluates them on a held-out future period, and
persists metrics, feature importances, and fitted pipelines.

Models
  1. Logistic Regression  — interpretable baseline; linear decision boundary.
  2. Random Forest        — non-linear ensemble; captures interaction effects.

Split strategy
  Time-based: rows are sorted by ETA and the last ML_TEST_SIZE fraction of
  completed calls forms the test set.  This mirrors production deployment:
  the model is trained on historical calls and scores future arrivals.

  Random splitting is intentionally avoided.  Rolling history features
  (avg_previous_10_terminal_delays etc.) are computed from the same call
  population used in training.  A random split would scatter future calls
  into the training set, whose delay outcomes would already be visible in
  those history features — leaking the future into training metrics.

Run:
    python src/train_model.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import warnings

import joblib
import numpy as np

# sklearn's solver internals emit numpy overflow RuntimeWarnings under
# class_weight="balanced" on certain NumPy / Python-3.9 builds.  Predictions
# remain valid — sklearn clips the log-loss gradient before it diverges.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    GOLD_DIR,
    ML_RANDOM_STATE,
    ML_TEST_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_PATH = GOLD_DIR / "ml_vessel_delay_features.parquet"
TARGET_COL    = "is_arrival_delayed_more_than_2_hours"

# Ordered list of model inputs.  The order matters: feature_importance.csv
# and pipeline internals both reference columns by position.
MODEL_FEATURES = [
    # Categorical (ordinal-encoded integers; stable mapping set in feature_engineering.py)
    "terminal_id_encoded",
    "service_code_encoded",
    # Static call attributes booked before vessel arrival
    "vessel_capacity_teu",
    "planned_moves",
    "planned_crane_count",
    "planned_crane_hours",
    # Temporal features derived from ETA (no actuals used)
    "day_of_week",
    "month",
    "is_weekend",
    # Weather observable as a forecast at ETA date
    "storm_flag",
    # Rolling history features (PIT: history.atd strictly < current.eta)
    "avg_previous_10_terminal_delays",
    "avg_previous_10_service_delays",
    "previous_vessel_delay",
    # Terminal congestion derived from planned ETA/ETD (no actuals used)
    "terminal_congestion_score",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_features() -> pd.DataFrame:
    """
    Load the ML feature table and discard rows without a known outcome.

    Unlabeled rows (target = NaN) are in-progress calls: valid inference
    targets at runtime, but excluded from training and test evaluation
    because the ground-truth label is unknown.
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURES_PATH}. "
            "Run src/feature_engineering.py first."
        )

    df = pd.read_parquet(FEATURES_PATH, engine="pyarrow")
    total = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    log.info("Loaded %d labeled rows  (%d unlabeled inference rows dropped)", len(df), total - len(df))
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def time_split(
    df: pd.DataFrame,
    test_size: float = ML_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Chronological train / test split keyed on ETA.

    The first (1 - test_size) fraction of rows by ETA forms training data;
    the remainder forms the test set.  Both sets are derived from the
    labeled pool only (unlabeled rows already dropped by load_features).

    Returns X_train, X_test, y_train, y_test, and a dict of split metadata
    written into model_metrics.json for reproducibility.
    """
    df = df.sort_values("eta").reset_index(drop=True)
    cutoff_idx  = int(len(df) * (1 - test_size))
    split_date  = df.iloc[cutoff_idx]["eta"]

    train = df.iloc[:cutoff_idx]
    test  = df.iloc[cutoff_idx:]

    X_train = train[MODEL_FEATURES]
    X_test  = test[MODEL_FEATURES]
    y_train = train[TARGET_COL].astype(int)
    y_test  = test[TARGET_COL].astype(int)

    split_info = {
        "strategy":            "time_based",
        "split_date":          split_date.isoformat(),
        "train_rows":          len(train),
        "test_rows":           len(test),
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test":  round(float(y_test.mean()),  4),
    }

    log.info(
        "Time split at %s  →  train=%d  test=%d  "
        "(delayed: train=%.1f%%  test=%.1f%%)",
        split_date.date(), len(train), len(test),
        100 * y_train.mean(), 100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test, split_info


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_lr_pipeline() -> Pipeline:
    """
    Logistic Regression pipeline: Imputer → Scaler → Classifier.

    SimpleImputer fills NaN values in rolling history features with the
    column median — robust to the right-skewed delay distribution.

    StandardScaler is required for Logistic Regression: L2 regularisation
    treats all coefficient magnitudes equally, so without scaling the
    optimiser penalises large-range features (vessel_capacity_teu, hours)
    disproportionately relative to binary flags (is_weekend, storm_flag).

    class_weight="balanced" up-weights the minority (delayed) class
    inversely proportional to its frequency, compensating for the 27 / 73
    positive / negative split.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
                        solver="liblinear",     # coordinate descent; stable predict_proba
                        max_iter=2000,
                        C=0.5,                  # moderate regularisation for small dataset
                        random_state=ML_RANDOM_STATE,
                        class_weight="balanced",
                    )),
    ])


def build_rf_pipeline() -> Pipeline:
    """
    Random Forest pipeline: Imputer → Classifier.

    Tree-based models split on thresholds, making them scale-invariant —
    StandardScaler is omitted.  SimpleImputer is still required because
    sklearn trees do not natively handle NaN.

    n_estimators=300 reduces variance; min_samples_leaf=10 prevents
    over-fitting on the ~1,500-row training set by ensuring every leaf
    represents at least 10 calls.  class_weight="balanced" mirrors the
    Logistic Regression treatment of the class imbalance.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        random_state=ML_RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    )),
    ])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Fit pipeline on train, score on test, return a metrics dictionary.

    Threshold-dependent metrics (accuracy, precision, recall, F1) use the
    default 0.5 decision threshold.  ROC-AUC uses predicted probabilities
    and is threshold-free — it measures discrimination ability across all
    thresholds, making it the primary comparison metric here.

    confusion_matrix is stored as [[TN, FP], [FN, TP]] — standard sklearn
    layout where rows are actual class and columns are predicted class.
    """
    log.info("Fitting %s ...", name)
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # P(delayed=1)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":          round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": cm.tolist(),
    }

    log.info(
        "%-22s  acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f",
        name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


def build_feature_importances(
    lr_pipeline: Pipeline,
    rf_pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Combine feature importances from both fitted models into one DataFrame.

    Logistic Regression: raw signed coefficient from the fitted model after
    StandardScaler, so magnitudes are on a comparable scale.  The normalised
    column shows |coefficient| / sum(|coefficients|) for ranking.

    Random Forest: mean decrease in Gini impurity across all trees.  The
    normalised column shows importance / sum(importances).

    Both normalised columns sum to 1.0, enabling apples-to-apples ranking
    across the two models regardless of their raw value scales.
    """
    lr_coef = lr_pipeline.named_steps["clf"].coef_[0]
    rf_imp  = rf_pipeline.named_steps["clf"].feature_importances_

    lr_abs  = np.abs(lr_coef)
    lr_norm = lr_abs / lr_abs.sum()
    rf_norm = rf_imp / rf_imp.sum()

    df = pd.DataFrame({
        "feature":            MODEL_FEATURES,
        "lr_coefficient":     np.round(lr_coef, 6),
        "lr_importance_norm": np.round(lr_norm, 6),
        "rf_importance":      np.round(rf_imp,  6),
        "rf_importance_norm": np.round(rf_norm, 6),
    })

    # Sort by RF importance descending — the more expressive model's ranking
    # is used as the canonical ordering in the output file.
    return df.sort_values("rf_importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_metrics(payload: dict) -> Path:
    """Write the full metrics payload to outputs/model_metrics.json."""
    out = OUTPUTS_DIR / "model_metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    log.info("Metrics → %s", out)
    return out


def save_feature_importances(df: pd.DataFrame) -> Path:
    """Write the feature importance table to outputs/feature_importance.csv."""
    out = OUTPUTS_DIR / "feature_importance.csv"
    df.to_csv(out, index=False)
    log.info("Feature importances → %s", out)
    return out


def save_model(pipeline: Pipeline, model_name: str) -> Path:
    """Persist a fitted sklearn Pipeline to models/<model_name>.joblib."""
    out = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(pipeline, out)
    log.info("Model → %s", out)
    return out


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def print_report(
    split_info:  dict,
    all_metrics: dict,
    imp_df:      pd.DataFrame,
) -> None:
    """Print a side-by-side model comparison and top feature rankings."""
    print()
    print(
        f"  Split      {split_info['split_date'][:10]}  "
        f"train={split_info['train_rows']:,}  test={split_info['test_rows']:,}  "
        f"|  delayed  train={100*split_info['positive_rate_train']:.1f}%  "
        f"test={100*split_info['positive_rate_test']:.1f}%"
    )
    print()

    # --- Metric table ---
    col_w = 25
    header = (
        f"  {'Model':<{col_w}}  {'Accuracy':>8}  {'Precision':>9}  "
        f"{'Recall':>8}  {'F1':>8}  {'ROC-AUC':>8}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for key, m in all_metrics.items():
        label = key.replace("_", " ").title()
        print(
            f"  {label:<{col_w}}  {m['accuracy']:>8.4f}  {m['precision']:>9.4f}  "
            f"{m['recall']:>8.4f}  {m['f1_score']:>8.4f}  {m['roc_auc']:>8.4f}"
        )
    print()

    # --- Confusion matrices ---
    for key, m in all_metrics.items():
        cm    = m["confusion_matrix"]
        label = key.replace("_", " ").title()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        print(f"  {label}  (rows=actual  cols=predicted)")
        print(f"    TN={tn:>4}  FP={fp:>4}   ← predicted on-time")
        print(f"    FN={fn:>4}  TP={tp:>4}   ← predicted delayed")
        print()

    # --- Top features ---
    print("  Top 5 features by Random Forest importance")
    print(f"  {'Feature':<42}  {'RF imp':>8}  {'LR coef':>9}")
    print("  " + "─" * 64)
    for _, row in imp_df.head(5).iterrows():
        print(
            f"  {row['feature']:<42}  {row['rf_importance']:>8.4f}  "
            f"{row['lr_coefficient']:>+9.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dirs()

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Model training  |  {ts}", flush=True)

    # --- Load ---
    df = load_features()

    # --- Split ---
    X_train, X_test, y_train, y_test, split_info = time_split(df)

    # --- Build pipelines ---
    lr_pipeline = build_lr_pipeline()
    rf_pipeline = build_rf_pipeline()

    # --- Train and evaluate ---
    lr_metrics = evaluate(
        "logistic_regression", lr_pipeline,
        X_train, y_train, X_test, y_test,
    )
    rf_metrics = evaluate(
        "random_forest", rf_pipeline,
        X_train, y_train, X_test, y_test,
    )

    # --- Persist models ---
    save_model(lr_pipeline, "logistic_regression")
    save_model(rf_pipeline, "random_forest")

    # --- Feature importances ---
    imp_df = build_feature_importances(lr_pipeline, rf_pipeline)
    save_feature_importances(imp_df)

    # --- Metrics JSON ---
    all_metrics = {
        "logistic_regression": lr_metrics,
        "random_forest":       rf_metrics,
    }
    payload = {
        "run_timestamp":  datetime.now(tz=timezone.utc).isoformat(),
        "feature_table":  str(FEATURES_PATH),
        "model_features": MODEL_FEATURES,
        "split":          split_info,
        "models":         all_metrics,
    }
    save_metrics(payload)

    # --- Console report ---
    print_report(split_info, all_metrics, imp_df)


if __name__ == "__main__":
    main()
