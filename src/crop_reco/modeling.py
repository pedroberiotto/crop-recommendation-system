from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.matplotlib-cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from .config import CV_FOLDS, RANDOM_STATE
from .preprocessing import build_pipeline

def get_candidate_models() -> dict[str, Any]:
    return {'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs'), 'gaussian_nb': GaussianNB(), 'random_forest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1), 'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)}

def build_full_pipeline(classifier: Any) -> Pipeline:
    return Pipeline([('preprocessor', build_pipeline()), ('classifier', clone(classifier))])

def run_cross_validation(X: pd.DataFrame, y: pd.Series, n_splits: int=CV_FOLDS) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    for name, clf in get_candidate_models().items():
        pipeline = build_full_pipeline(clf)
        try:
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=['accuracy', 'f1_macro'], n_jobs=1, error_score='raise')
            rows.append({'model': name, 'accuracy_mean': float(np.mean(scores['test_accuracy'])), 'accuracy_std': float(np.std(scores['test_accuracy'])), 'f1_macro_mean': float(np.mean(scores['test_f1_macro'])), 'f1_macro_std': float(np.std(scores['test_f1_macro'])), 'error': ''})
        except Exception as exc:
            rows.append({'model': name, 'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'f1_macro_mean': np.nan, 'f1_macro_std': np.nan, 'error': f'{type(exc).__name__}: {exc}'})
    result = pd.DataFrame(rows)
    if result['f1_macro_mean'].notna().sum() == 0:
        raise RuntimeError('No candidate model completed cross-validation.')
    return result.sort_values('f1_macro_mean', ascending=False, na_position='last').reset_index(drop=True)

def train_final_model(X: pd.DataFrame, y: pd.Series, clf: Any) -> Pipeline:
    pipeline = build_full_pipeline(clf)
    pipeline.fit(X, y)
    return pipeline

def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, figures_dir: Optional[Path]=None) -> dict:
    y_pred = model.predict(X)
    classes = sorted(y.unique())
    report_str = classification_report(y, y_pred, labels=classes, target_names=classes, zero_division=0)
    report_dict = classification_report(y, y_pred, labels=classes, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=classes)
    result: dict = {'accuracy': report_dict['accuracy'], 'f1_macro': report_dict['macro avg']['f1-score'], 'report_str': report_str, 'report_dict': report_dict, 'confusion_matrix': cm, 'classes': classes, 'paths': {}}
    if figures_dir is not None:
        fdir = Path(figures_dir)
        result['paths'] = {'confusion_matrix': save_confusion_matrix(cm, classes, fdir / 'confusion_matrix.png'), 'classification_report': save_classification_report(report_str, fdir / 'classification_report.txt')}
    return result

def select_best(cv_df: pd.DataFrame) -> str:
    valid = cv_df.dropna(subset=['f1_macro_mean'])
    if valid.empty:
        raise ValueError('cv_df has no valid models for selection.')
    return str(valid.iloc[0]['model'])

def save_confusion_matrix(cm, classes, path, title='Confusion Matrix — test set'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.3, linecolor='white')
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path

def save_cv_comparison(cv_df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = cv_df.dropna(subset=['f1_macro_mean'])
    names = valid['model'].tolist()
    means = valid['f1_macro_mean'].tolist()
    stds = valid['f1_macro_std'].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, means, xerr=stds, capsize=5)
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.002, bar.get_y() + bar.get_height() / 2, f'{mean:.4f}', va='center', fontsize=9)
    ax.set_xlabel('macro F1 (mean ± std, 5-fold CV)')
    ax.set_title('Model comparison — stratified cross-validation')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path

def save_classification_report(report_str: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_str, encoding='utf-8')
    return path
