from __future__ import annotations
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import sklearn
from crop_reco.config import ARTIFACTS_DIR, NUMERIC_FEATURES
from crop_reco.logger import log_event
logger = logging.getLogger(__name__)

class ModelStore:

    def __init__(self, artifacts_dir: Optional[Path]=None) -> None:
        self._artifacts_dir: Path = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self._model = None
        self._artifact_name: Optional[str] = None
        self._loaded_at: Optional[str] = None
        self._serialized_at: Optional[str] = None
        self._sklearn_version_train: Optional[str] = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def _artifact_path(self, artifact_name: str) -> Path:
        if Path(artifact_name).name != artifact_name or not artifact_name.endswith('.pkl'):
            raise ValueError('artifact_name must be a plain .pkl filename')
        return self._artifacts_dir / artifact_name

    def load(self, artifact_name: str='final_model.pkl') -> dict:
        path = self._artifact_path(artifact_name)
        if not path.exists():
            available = [p.name for p in self._artifacts_dir.glob('*.pkl')]
            raise FileNotFoundError(f'Artifact not found: {path}\nAvailable artifacts: {available}')
        try:
            candidate = joblib.load(path)
        except Exception as exc:
            raise ValueError(f'Artifact is not deserializable: {exc}') from exc
        self._validate_model(candidate)
        meta = self._read_sidecar_metadata(path)
        train_version = meta.get('sklearn_version') or self._extract_sklearn_version(candidate)
        runtime_version = sklearn.__version__
        version_mismatch = bool(train_version and train_version != runtime_version)
        if version_mismatch:
            log_event('sklearn_version_mismatch', artifact=artifact_name, version_train=train_version, version_runtime=runtime_version)
            logger.warning('sklearn version mismatch: train=%s runtime=%s — predictions may silently diverge.', train_version, runtime_version)
        with self._lock:
            self._model = candidate
            self._artifact_name = artifact_name
            self._loaded_at = datetime.now(timezone.utc).isoformat()
            self._serialized_at = meta.get('serialized_at')
            self._sklearn_version_train = train_version
            info = self._info_unlocked()
        log_event('model_loaded', artifact=artifact_name, sklearn_version_train=train_version, sklearn_version_runtime=runtime_version, version_mismatch=version_mismatch)
        return info

    def _read_sidecar_metadata(self, artifact_path: Path) -> dict:
        candidates = [artifact_path.with_name(f'{artifact_path.stem}_meta.json'), artifact_path.with_suffix('.json')]
        for meta_path in candidates:
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding='utf-8'))
                    return data if isinstance(data, dict) else {}
                except Exception as exc:
                    logger.warning('Could not read metadata from artifact %s: %s', meta_path, exc)
        return {}

    def _validate_model(self, model) -> None:
        missing = [name for name in ('predict', 'predict_proba') if not hasattr(model, name)]
        if missing:
            raise ValueError(f'Artifact does not appear to be a valid sklearn model; missing methods: {missing}')
        if not hasattr(model, 'classes_'):
            raise ValueError('Invalid artifact: classes_ attribute missing')
        if hasattr(model, 'named_steps'):
            required_steps = {'preprocessor', 'classifier'}
            missing_steps = required_steps - set(model.named_steps.keys())
            if missing_steps:
                raise ValueError(f'Invalid pipeline; missing steps: {sorted(missing_steps)}')

    def _extract_sklearn_version(self, model) -> Optional[str]:
        try:
            clf = model.named_steps.get('classifier', model)
            if hasattr(clf, '_sklearn_version'):
                return str(clf._sklearn_version)
            state = clf.__getstate__() if hasattr(clf, '__getstate__') else {}
            if isinstance(state, dict) and '_sklearn_version' in state:
                return str(state['_sklearn_version'])
        except Exception:
            pass
        return None

    def get_model(self):
        with self._lock:
            if self._model is None:
                raise RuntimeError('No model loaded.')
            return self._model

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        with self._lock:
            if self._model is None:
                raise RuntimeError('No model loaded.')
            return self._model.predict_proba(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        with self._lock:
            if self._model is None:
                raise RuntimeError('No model loaded.')
            return self._model.predict(df)

    def classes(self) -> list[str]:
        with self._lock:
            if self._model is None:
                raise RuntimeError('No model loaded.')
            return list(self._model.classes_)

    def rollback(self, artifact_name: str) -> dict:
        previous = self._artifact_name
        info = self.load(artifact_name)
        log_event('model_rollback', from_artifact=previous, to_artifact=artifact_name)
        return info

    def _info_unlocked(self) -> dict:
        return {'artifact': self._artifact_name, 'loaded_at': self._loaded_at, 'serialized_at': self._serialized_at, 'sklearn_version_train': self._sklearn_version_train, 'sklearn_version_runtime': sklearn.__version__, 'features': NUMERIC_FEATURES, 'n_classes': len(self._model.classes_) if self._model is not None else 0}

    def info(self) -> dict:
        with self._lock:
            return self._info_unlocked()

    def available_artifacts(self) -> list[str]:
        return sorted((p.name for p in self._artifacts_dir.glob('*.pkl')))
model_store = ModelStore()
