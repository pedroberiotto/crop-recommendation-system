from __future__ import annotations
import hashlib
import json
import pytest
from crop_reco.config import ARTIFACTS_DIR, MODEL_FINAL_PATH, MODEL_META_PATH
pytestmark = pytest.mark.skipif(not MODEL_FINAL_PATH.exists(), reason='final_model.pkl not found — run `make train` first')

class TestArtifactCompat:

    def test_artifact_loadable(self):
        import joblib
        model = joblib.load(MODEL_FINAL_PATH)
        assert model is not None

    def test_metadata_exists(self):
        assert MODEL_META_PATH.exists(), f'{MODEL_META_PATH.name} not found. The training script always writes the sidecar; its absence indicates a corrupted training run.'

    def test_sklearn_version_compatible(self):
        import sklearn
        meta = json.loads(MODEL_META_PATH.read_text(encoding='utf-8'))
        train_v = meta.get('sklearn_version')
        runtime_v = sklearn.__version__
        assert train_v is not None, f"field 'sklearn_version' missing in {MODEL_META_PATH.name}"
        assert train_v == runtime_v, f'sklearn mismatch: artifact={train_v} runtime={runtime_v}. Use POST /model/rollback to revert to a compatible version or retrain with the current runtime version.'

    def test_pipeline_has_preprocessor_and_classifier(self):
        import joblib
        model = joblib.load(MODEL_FINAL_PATH)
        assert 'preprocessor' in model.named_steps
        assert 'classifier' in model.named_steps

    def test_predicts_22_classes(self):
        import joblib
        model = joblib.load(MODEL_FINAL_PATH)
        assert len(model.classes_) == 22

    def test_predict_proba_on_sample(self):
        import joblib
        import numpy as np
        import pandas as pd
        model = joblib.load(MODEL_FINAL_PATH)
        sample = pd.DataFrame([{'N': 90, 'P': 42, 'K': 43, 'temperature': 20.9, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9}, {'N': 20, 'P': 30, 'K': 10, 'temperature': 27.0, 'humidity': 65.0, 'ph': 7.0, 'rainfall': 100.0}, {'N': 40, 'P': 80, 'K': 40, 'temperature': 15.0, 'humidity': 90.0, 'ph': 5.5, 'rainfall': 250.0}])
        proba = model.predict_proba(sample)
        assert proba.shape == (3, 22)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-05)
        assert (proba >= 0).all()

    def test_model_v1_exists_and_is_different(self):
        v1 = ARTIFACTS_DIR / 'model_v1.pkl'
        if not v1.exists():
            pytest.skip('model_v1.pkl not found — run `make train` first')
        md5_final = hashlib.md5(MODEL_FINAL_PATH.read_bytes()).hexdigest()
        md5_v1 = hashlib.md5(v1.read_bytes()).hexdigest()
        assert md5_final != md5_v1, 'final_model.pkl and model_v1.pkl have the same hash — the rollback path does not exercise a real model switch.'
