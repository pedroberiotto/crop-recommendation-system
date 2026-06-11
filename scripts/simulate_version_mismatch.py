from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import sklearn
from crop_reco.config import ARTIFACTS_DIR, MODEL_PATH
FAKE_VERSION = '1.3.0'

def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f'final_model.pkl not found at {MODEL_PATH}. Run `make train` first.')
    runtime_version = sklearn.__version__
    fake = FAKE_VERSION if FAKE_VERSION != runtime_version else '1.2.99'
    pipeline = joblib.load(MODEL_PATH)
    import sklearn.base as _sklbase
    original_version = sklearn.__version__
    original_base = _sklbase.__version__
    sklearn.__version__ = fake
    _sklbase.__version__ = fake
    try:
        legacy_path = ARTIFACTS_DIR / 'model_v_legacy.pkl'
        joblib.dump(pipeline, legacy_path)
    finally:
        sklearn.__version__ = original_version
        _sklbase.__version__ = original_base
    reloaded = joblib.load(legacy_path)
    clf = reloaded.named_steps['classifier']
    state = clf.__getstate__() if hasattr(clf, '__getstate__') else {}
    recorded = state.get('_sklearn_version') if isinstance(state, dict) else None
    import json
    from datetime import datetime, timezone
    sidecar = legacy_path.with_name(f'{legacy_path.stem}_meta.json')
    sidecar.write_text(json.dumps({'model_name': 'random_forest', 'sklearn_version': fake, 'serialized_at': datetime.now(timezone.utc).isoformat(), 'mlflow_run_id': 'simulated-legacy', 'artifact_path': f'artifacts/{legacy_path.name}'}, indent=2), encoding='utf-8')
    print(f'✓ model_v_legacy.pkl generated')
    print(f'  version recorded in pickle (via __getstate__): {recorded!r}')
    print(f'  version recorded in sidecar JSON: {fake!r}')
    print(f'  current runtime:                              {runtime_version!r}')
    print()
    print('Note: when deserializing, sklearn POPs `_sklearn_version` in')
    print('__setstate__ — so the attribute does not survive the reload. The')
    print('ModelStore diagnostic relies on the sidecar JSON, which records')
    print('the real version at serialization time.')
    print()
    print('To trigger the version mismatch diagnostic via the API:')
    print('  curl -X POST http://localhost:8000/model/rollback \\')
    print("       -H 'Content-Type: application/json' \\")
    print('       -d \'{"artifact": "model_v_legacy.pkl"}\'')
    print()
    print("The API JSON log should include a 'sklearn_version_mismatch' event")
    print(f'with version_train={fake!r} and version_runtime={runtime_version!r}.')
if __name__ == '__main__':
    main()
