"""
Model training, evaluation, comparison, serialization, and inference module.
Trains multiple classifiers (Logistic Regression, Random Forest, XGBoost),
computes comprehensive clinical evaluation metrics, serializes the best pipeline,
and provides a reusable risk prediction function.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from src.data_prep import get_prepared_data, FEATURE_NAMES, clean_physiological_zeros


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classification models with tuned hyperparameters."""
    return {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=settings.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=settings.RANDOM_STATE,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=1.8,
            eval_metric="logloss",
            random_state=settings.RANDOM_STATE,
        ),
    }


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Calculate core classification metrics on test data."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    return metrics


def train_and_compare() -> Tuple[Dict[str, Any], str, Pipeline]:
    """
    Train each candidate model inside a clean preprocessing pipeline,
    record comparison metrics, select the optimal model for clinical screening,
    and save reports and artifacts.
    """
    X_train, X_test, y_train, y_test, preprocessor = get_prepared_data()
    models = get_candidate_models()

    results = {}
    best_model_name = ""
    best_score = -1.0
    best_pipeline = None

    print("\n" + "=" * 75)
    print("HEALTHCARE ML MODEL COMPARISON & BENCHMARKING")
    print("=" * 75)

    for name, clf in models.items():
        # Build unified leak-free pipeline
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = metrics

        # Composite clinical score prioritizing ROC-AUC and Recall (minimizing false negatives)
        composite_score = 0.55 * metrics["roc_auc"] + 0.45 * metrics["recall"]
        results[name]["composite_clinical_score"] = round(composite_score, 4)

        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name
            best_pipeline = pipe

    # Save metrics JSON report
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "best_model": best_model_name,
        "selection_rationale": (
            "Selected based on balanced ROC-AUC and Recall. In clinical screening, false negatives "
            "(undiagnosed high-risk patients) pose substantial health risks, making high Recall "
            "and strong discriminative ability (ROC-AUC) paramount."
        ),
        "models": results,
    }
    with open(settings.METRICS_PATH, "w") as f:
        json.dump(report_data, f, indent=2)

    # Save best serialized model pipeline
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, settings.MODEL_PATH)

    # Console display table
    print(f"{'Model Name':<35} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 75)
    for name, m in results.items():
        print(
            f"{name:<35} | {m['accuracy']:<6.4f} | {m['precision']:<6.4f} | "
            f"{m['recall']:<6.4f} | {m['f1_score']:<6.4f} | {m['roc_auc']:<7.4f}"
        )
    print("-" * 75)
    print(f"\nWinning Model Selected: {best_model_name} (Composite Clinical Score: {best_score:.4f})")
    print(f"Artifact serialized to: {settings.MODEL_PATH}")
    print(f"Metrics saved to:       {settings.METRICS_PATH}\n")

    return report_data, best_model_name, best_pipeline


def load_inference_pipeline() -> Pipeline:
    """Load persisted production pipeline from disk."""
    if not settings.MODEL_PATH.exists():
        print("Model file not found. Training model now...")
        train_and_compare()
    return joblib.load(settings.MODEL_PATH)


def determine_risk_tier(prob: float) -> Tuple[str, str]:
    """Map probability to clinical risk tier and color representation."""
    if prob < 0.25:
        return "Low Risk", "#28a745"
    elif prob < 0.50:
        return "Moderate Risk", "#ffc107"
    elif prob < 0.75:
        return "High Risk", "#fd7e14"
    else:
        return "Critical Risk", "#dc3545"


def predict_patient_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reusable inference function for scoring a single patient's clinical parameters.
    
    Parameters:
        patient_data: dict containing keys:
            - Pregnancies (int)
            - Glucose (float)
            - BloodPressure (float)
            - SkinThickness (float)
            - Insulin (float)
            - BMI (float)
            - DiabetesPedigreeFunction (float)
            - Age (int/float)
            
    Returns:
        dict containing predicted class, risk probability, risk tier, and key clinical flags.
    """
    pipeline = load_inference_pipeline()
    
    # Format input into single-row DataFrame
    row = {k: [patient_data.get(k, np.nan)] for k in FEATURE_NAMES}
    df_input = pd.DataFrame(row)
    
    # Flag physiological zeros as NaN so imputer acts consistently
    df_cleaned = clean_physiological_zeros(df_input)
    
    # Predict
    prob = float(pipeline.predict_proba(df_cleaned)[0, 1])
    pred_class = int(prob >= 0.5)
    risk_tier, tier_color = determine_risk_tier(prob)

    # Clinical risk flags identification
    clinical_flags = []
    glucose_val = patient_data.get("Glucose", 0)
    bp_val = patient_data.get("BloodPressure", 0)
    bmi_val = patient_data.get("BMI", 0)
    age_val = patient_data.get("Age", 0)

    if glucose_val >= 126:
        clinical_flags.append(f"Fasting Glucose {glucose_val} mg/dL meets diabetic threshold (>= 126 mg/dL).")
    elif glucose_val >= 100:
        clinical_flags.append(f"Fasting Glucose {glucose_val} mg/dL indicates impaired fasting glucose / prediabetes (100-125 mg/dL).")
        
    if bp_val >= 140:
        clinical_flags.append(f"Blood pressure {bp_val} mmHg is in Stage 2 Hypertension range (>= 140 mmHg).")
    elif bp_val >= 130:
        clinical_flags.append(f"Blood pressure {bp_val} mmHg is in Stage 1 Hypertension range (130-139 mmHg).")
        
    if bmi_val >= 30:
        clinical_flags.append(f"BMI of {bmi_val:.1f} kg/m² indicates Obesity class I/II (>= 30 kg/m²).")
    elif bmi_val >= 25:
        clinical_flags.append(f"BMI of {bmi_val:.1f} kg/m² indicates Overweight (25.0-29.9 kg/m²).")

    if age_val >= 45:
        clinical_flags.append(f"Age {int(age_val)} is in the elevated ADA screening risk tier (>= 45 years).")

    return {
        "predicted_class": pred_class,
        "predicted_label": "High Risk" if pred_class == 1 else "Low Risk",
        "risk_probability": round(prob, 4),
        "risk_percentage": round(prob * 100, 1),
        "risk_tier": risk_tier,
        "risk_color": tier_color,
        "clinical_flags": clinical_flags,
        "patient_data": patient_data,
    }


if __name__ == "__main__":
    train_and_compare()
