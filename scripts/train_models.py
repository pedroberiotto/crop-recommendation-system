from __future__ import annotations
import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import joblib
import sklearn
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
from crop_reco.config import ARTIFACTS_DIR, CV_FOLDS, FIGURES_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI_DEFAULT, MODEL_META_PATH, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from crop_reco.data import load_dataset, split_xy
from crop_reco.modeling import evaluate_model, get_candidate_models, run_cross_validation, save_cv_comparison, select_best, train_final_model
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

class _DummyRunInfo:
    run_id = 'mlflow-disabled'

class _DummyRun:
    info = _DummyRunInfo()

@contextmanager
def _start_run(run_name: str):
    if mlflow is None:
        yield _DummyRun()
    else:
        with mlflow.start_run(run_name=run_name) as run:
            yield run

def _mlflow_log_param(*args, **kwargs) -> None:
    if mlflow is not None:
        mlflow.log_param(*args, **kwargs)

def _mlflow_log_metric(*args, **kwargs) -> None:
    if mlflow is not None:
        mlflow.log_metric(*args, **kwargs)

def _mlflow_set_tag(*args, **kwargs) -> None:
    if mlflow is not None:
        mlflow.set_tag(*args, **kwargs)

def _mlflow_log_artifact(*args, **kwargs) -> None:
    if mlflow is not None:
        mlflow.log_artifact(*args, **kwargs)

def setup_mlflow() -> None:
    if mlflow is None:
        log.warning('MLflow is not installed; training will proceed without remote/local tracking.')
        return
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', MLFLOW_TRACKING_URI_DEFAULT)
    if tracking_uri.startswith('sqlite:///'):
        db_path = Path(tracking_uri.replace('sqlite:///', ''))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    log.info('MLflow tracking URI: %s', tracking_uri)

def _clean_orphan_sidecars() -> int:
    removed = 0
    for json_path in ARTIFACTS_DIR.glob('*.json'):
        pkl_path = json_path.with_suffix('.pkl')
        if json_path.stem.endswith('_meta'):
            pkl_path = json_path.with_name(json_path.stem[:-5] + '.pkl')
        if not pkl_path.exists():
            json_path.unlink()
            log.info('Orphan sidecar removed: %s', json_path.name)
            removed += 1
    return removed

def _backup_previous_model() -> Path | None:
    if not MODEL_PATH.exists():
        return None
    n = 1
    while (ARTIFACTS_DIR / f'model_v{n}.pkl').exists():
        n += 1
    backup = ARTIFACTS_DIR / f'model_v{n}.pkl'
    shutil.copy2(MODEL_PATH, backup)
    if MODEL_META_PATH.exists():
        shutil.copy2(MODEL_META_PATH, backup.with_suffix('.json'))
    log.info('Previous model backup: %s', backup.name)
    return backup

def _write_model_meta(model_name: str, run_id: str) -> None:
    try:
        relative_artifact_path = MODEL_PATH.relative_to(MODEL_PATH.parents[1])
    except ValueError:
        relative_artifact_path = MODEL_PATH.name
    meta = {'model_name': model_name, 'sklearn_version': sklearn.__version__, 'serialized_at': datetime.now(timezone.utc).isoformat(), 'mlflow_run_id': run_id, 'artifact_path': str(relative_artifact_path)}
    MODEL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    log.info('Metadata saved to %s', MODEL_META_PATH)

def _write_confusion_analysis(test_metrics: dict, model_name: str) -> Path:
    cm = test_metrics['confusion_matrix']
    classes = test_metrics['classes']
    accuracy = test_metrics['accuracy']
    total = int(cm.sum())
    errors_total = int(total - cm.trace())
    error_pairs = []
    for i, real in enumerate(classes):
        for j, predicted in enumerate(classes):
            if i != j and cm[i][j] > 0:
                error_pairs.append((real, predicted, int(cm[i][j])))
    error_pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    per_class_errors: dict[str, int] = {}
    for real, _, count in error_pairs:
        per_class_errors[real] = per_class_errors.get(real, 0) + count
    absorbing_classes: dict[str, int] = {}
    for _, predicted, count in error_pairs:
        absorbing_classes[predicted] = absorbing_classes.get(predicted, 0) + count
    lines = []
    lines.append('# Test set error analysis\n')
    lines.append('> File generated automatically by `scripts/train_models.py` from the real confusion matrix.\n')
    lines.append(f'- **Model:** `{model_name}`')
    lines.append(f'- **Accuracy:** {accuracy:.4f}')
    lines.append(f'- **Total samples:** {total}')
    lines.append(f'- **Total errors:** {errors_total}')
    lines.append(f'- **Generated at:** {datetime.now(timezone.utc).isoformat()}\n')
    lines.append('## Pairs (actual → predicted)\n')
    if not error_pairs:
        lines.append('No errors on the test set — 100% accuracy.\n')
    else:
        lines.append('| Actual crop | Predicted crop | No. of errors |')
        lines.append('|---|---|---:|')
        for real, predicted, count in error_pairs:
            lines.append(f'| `{real}` | `{predicted}` | {count} |')
        lines.append('')
    if per_class_errors:
        lines.append('## Errors per actual crop (recall < 1)\n')
        lines.append('| Crop | Errors |')
        lines.append('|---|---:|')
        for crop, count in sorted(per_class_errors.items(), key=lambda x: -x[1]):
            lines.append(f'| `{crop}` | {count} |')
        lines.append('')
    if absorbing_classes:
        lines.append('## Absorbing crops (precision < 1)\n')
        lines.append('| Crop | False positives |')
        lines.append('|---|---:|')
        for crop, count in sorted(absorbing_classes.items(), key=lambda x: -x[1]):
            lines.append(f'| `{crop}` | {count} |')
        lines.append('')
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / 'real_confusion_analysis.md'
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    log.info('Error analysis saved to %s', out_path)
    return out_path

def main() -> None:
    setup_mlflow()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _clean_orphan_sidecars()
    log.info('==> 1/6  Loading dataset')
    df = load_dataset()
    X, y = split_xy(df)
    log.info('     shape=%s | classes=%d', df.shape, y.nunique())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log.info('     train=%d | test=%d', len(X_train), len(X_test))
    log.info('==> 2/6  Cross-validation (%d folds) for candidates', CV_FOLDS)
    cv_df = run_cross_validation(X_train, y_train, n_splits=CV_FOLDS)
    cv_results_path = FIGURES_DIR.parent / 'cv_results.csv'
    cv_df.to_csv(cv_results_path, index=False)
    log.info('==> CV results saved to %s', cv_results_path)
    log.info('\n%s', cv_df.to_string(index=False))
    if 'error' in cv_df.columns and cv_df['error'].astype(bool).any():
        log.warning('Some candidates failed and were ignored during selection:\n%s', cv_df.loc[cv_df['error'].astype(bool), ['model', 'error']].to_string(index=False))
    log.info('==> 3/6  Registering runs in MLflow')
    candidates = get_candidate_models()
    for _, row in cv_df.iterrows():
        name = row['model']
        clf = candidates.get(name)
        with _start_run(str(name)):
            _mlflow_log_param('model_name', name)
            _mlflow_log_param('cv_folds', CV_FOLDS)
            _mlflow_log_param('test_size', TEST_SIZE)
            _mlflow_log_param('sklearn_version', sklearn.__version__)
            if clf is not None:
                for k, v in clf.get_params(deep=False).items():
                    try:
                        _mlflow_log_param(f'clf_{k}', v)
                    except Exception:
                        pass
            for metric_name in ('accuracy_mean', 'accuracy_std', 'f1_macro_mean', 'f1_macro_std'):
                if not row.isna()[metric_name]:
                    _mlflow_log_metric(f'cv_{metric_name}', float(row[metric_name]))
            _mlflow_set_tag('sklearn_version', sklearn.__version__)
            if row.get('error'):
                _mlflow_set_tag('cv_error', row['error'])
    cv_fig = save_cv_comparison(cv_df, FIGURES_DIR / 'cv_model_comparison.png')
    log.info('==> figura comparativa: %s', cv_fig)
    best_name = select_best(cv_df)
    best_score = float(cv_df.dropna(subset=['f1_macro_mean']).iloc[0]['f1_macro_mean'])
    log.info('==> 4/6  Best model: %s (F1_macro=%.4f)', best_name, best_score)
    best_clf = candidates[best_name]
    best_pipeline = train_final_model(X_train, y_train, best_clf)
    log.info('==> 5/6  Evaluating on test set')
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, figures_dir=FIGURES_DIR)
    log.info('\n%s', test_metrics['report_str'])
    confusion_md = _write_confusion_analysis(test_metrics, best_name)
    log.info('==> 6/6  Serializing final_model.pkl')
    backup = _backup_previous_model()
    with _start_run(f'{best_name}_final') as run:
        _mlflow_log_param('model_name', best_name)
        _mlflow_log_param('sklearn_version', sklearn.__version__)
        _mlflow_log_metric('test_accuracy', test_metrics['accuracy'])
        _mlflow_log_metric('test_f1_macro', test_metrics['f1_macro'])
        _mlflow_set_tag('sklearn_version', sklearn.__version__)
        _mlflow_set_tag('best_model', best_name)
        for path in test_metrics['paths'].values():
            _mlflow_log_artifact(str(path))
        _mlflow_log_artifact(str(cv_fig))
        _mlflow_log_artifact(str(confusion_md))
        if mlflow is not None:
            mlflow.sklearn.log_model(best_pipeline, artifact_path='model', registered_model_name=MLFLOW_MODEL_NAME)
        run_id = run.info.run_id
    joblib.dump(best_pipeline, MODEL_PATH)
    _write_model_meta(best_name, run_id)
    log.info('✓ final_model.pkl saved to %s', MODEL_PATH)
    if backup:
        log.info('  Backup for rollback: %s', backup.name)
    v1_path = ARTIFACTS_DIR / 'model_v1.pkl'
    if not v1_path.exists():
        ranked = cv_df.dropna(subset=['f1_macro_mean']).reset_index(drop=True)
        if len(ranked) >= 2:
            second_name = str(ranked.iloc[1]['model'])
            log.info('==> bootstrap rollback: serializing model_v1.pkl (%s)', second_name)
            second_pipeline = train_final_model(X_train, y_train, candidates[second_name])
            joblib.dump(second_pipeline, v1_path)
            try:
                rel_v1 = v1_path.relative_to(v1_path.parents[1])
            except ValueError:
                rel_v1 = v1_path.name
            v1_meta = {'model_name': second_name, 'sklearn_version': sklearn.__version__, 'serialized_at': datetime.now(timezone.utc).isoformat(), 'mlflow_run_id': 'bootstrap', 'artifact_path': str(rel_v1)}
            (ARTIFACTS_DIR / 'model_v1.json').write_text(json.dumps(v1_meta, indent=2), encoding='utf-8')
    loaded = joblib.load(MODEL_PATH)
    check_out = loaded.predict_proba(X_test.head(3))
    assert check_out.shape == (3, y.nunique()), 'Sanity check failed'
    log.info('✓ Sanity check OK — predict_proba shape: %s', check_out.shape)
if __name__ == '__main__':
    main()
