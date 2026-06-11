from __future__ import annotations
from pathlib import Path
import pytest

class TestHealth:

    def test_returns_200_with_model_loaded(self, api_client):
        r = api_client.get('/health')
        assert r.status_code == 200

    def test_body_contains_status_ok(self, api_client):
        body = api_client.get('/health').json()
        assert body['status'] == 'ok'
        assert body['model_loaded'] is True

    def test_returns_503_without_model(self):
        import app.model_store as ms_mod
        from app.main import app
        from fastapi.testclient import TestClient
        original_dir = ms_mod.model_store._artifacts_dir
        original_model = ms_mod.model_store._model
        ms_mod.model_store._artifacts_dir = Path('/tmp/no_artifacts_xyz_test_abc')
        ms_mod.model_store._model = None
        try:
            with TestClient(app) as client:
                r = client.get('/health')
            assert r.status_code == 503
            assert r.json()['model_loaded'] is False
        finally:
            ms_mod.model_store._artifacts_dir = original_dir
            ms_mod.model_store._model = original_model

class TestModelInfo:

    def test_returns_200(self, api_client):
        r = api_client.get('/model/info')
        assert r.status_code == 200

    def test_required_fields(self, api_client):
        body = api_client.get('/model/info').json()
        required = {'artifact', 'loaded_at', 'sklearn_version_runtime', 'features', 'n_classes', 'available_artifacts'}
        assert required.issubset(body.keys())

    def test_correct_features(self, api_client):
        from crop_reco.config import NUMERIC_FEATURES
        body = api_client.get('/model/info').json()
        assert body['features'] == NUMERIC_FEATURES

    def test_n_classes_equals_22(self, api_client):
        body = api_client.get('/model/info').json()
        assert body['n_classes'] == 22

    def test_sklearn_version_present(self, api_client):
        body = api_client.get('/model/info').json()
        assert isinstance(body['sklearn_version_runtime'], str)
        assert len(body['sklearn_version_runtime']) > 0
