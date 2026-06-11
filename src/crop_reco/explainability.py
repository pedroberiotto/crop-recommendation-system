from __future__ import annotations
import weakref
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from .config import FINAL_FEATURES
_EXPLAINER_CACHE: 'weakref.WeakKeyDictionary[object, object]' = weakref.WeakKeyDictionary()
_EPSILON = 0.001

def _compute_all_values(X_raw: pd.DataFrame) -> dict[str, float]:
    row = X_raw.iloc[0]
    N = float(row['N'])
    P = float(row['P'])
    K = float(row['K'])
    return {'N': N, 'P': P, 'K': K, 'temperature': float(row['temperature']), 'humidity': float(row['humidity']), 'ph': float(row['ph']), 'rainfall': float(row['rainfall']), 'NP_ratio': N / (P + _EPSILON), 'NK_ratio': N / (K + _EPSILON), 'PK_ratio': P / (K + _EPSILON), 'NPK_sum': N + P + K}

def _shap_values_for_class(shap_values, class_idx: int, sample_idx: int=0) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.array(shap_values[class_idx][sample_idx])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[sample_idx, :, class_idx]
    return shap_values[sample_idx]

def _unwrap_classifier(classifier):
    if type(classifier).__name__ == 'LabelEncodedClassifier' and hasattr(classifier, 'estimator_'):
        return classifier.estimator_
    return classifier

def _get_explainer(classifier, X_transformed):
    try:
        return _EXPLAINER_CACHE[classifier]
    except Exception:
        pass
    import shap
    base_classifier = _unwrap_classifier(classifier)
    clf_type = type(base_classifier).__name__
    if clf_type in ('RandomForestClassifier', 'GradientBoostingClassifier'):
        explainer = shap.TreeExplainer(base_classifier)
    elif clf_type == 'LogisticRegression':
        explainer = shap.LinearExplainer(base_classifier, X_transformed)
    else:
        return None
    try:
        _EXPLAINER_CACHE[classifier] = explainer
    except Exception:
        pass
    return explainer

def explain_predictions(model: Pipeline, X_raw: pd.DataFrame, predicted_classes: list[str], classes: list[str]) -> list[list[dict] | None]:
    if X_raw.empty:
        return []
    try:
        import shap
    except ImportError:
        return [None] * len(X_raw)
    try:
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        if type(_unwrap_classifier(classifier)).__name__ in ('GaussianNB', 'KNeighborsClassifier'):
            return [None] * len(X_raw)
        X_transformed = preprocessor.transform(X_raw)
        explainer = _get_explainer(classifier, X_transformed)
        if explainer is None:
            return [None] * len(X_raw)
        shap_vals = explainer.shap_values(X_transformed)
        feature_names = list(FINAL_FEATURES)
        explanations: list[list[dict] | None] = []
        for sample_idx, predicted_class in enumerate(predicted_classes):
            class_idx = list(classes).index(predicted_class)
            values = _shap_values_for_class(shap_vals, class_idx, sample_idx=sample_idx)
            feature_values = _compute_all_values(X_raw.iloc[[sample_idx]])
            top_idx = np.argsort(np.abs(values))[::-1][:3]
            explanations.append([{'feature': feature_names[i], 'value': round(feature_values[feature_names[i]], 4), 'impact': 'positive' if values[i] > 0 else 'negative'} for i in top_idx])
        return explanations
    except Exception:
        return [None] * len(X_raw)

def explain_prediction(model: Pipeline, X_raw: pd.DataFrame, predicted_class: str, classes: list[str]) -> list[dict] | None:
    explanations = explain_predictions(model, X_raw, [predicted_class], classes)
    return explanations[0] if explanations else None
