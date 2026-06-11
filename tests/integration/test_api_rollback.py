from __future__ import annotations
import pytest

class TestValidRollback:

    def test_rollback_to_v1_returns_200(self, rollback_client):
        r = rollback_client.post('/model/rollback', json={'artifact': 'model_v1.pkl'})
        assert r.status_code == 200

    def test_status_ok(self, rollback_client):
        r = rollback_client.post('/model/rollback', json={'artifact': 'model_v1.pkl'})
        assert r.json()['status'] == 'ok'

    def test_returns_active_model_metadata(self, rollback_client):
        r = rollback_client.post('/model/rollback', json={'artifact': 'model_v1.pkl'})
        body = r.json()
        assert 'active_model' in body
        info = body['active_model']
        assert info['artifact'] == 'model_v1.pkl'

    def test_switched_model_predicts_correctly(self, rollback_client, valid_payload):
        rollback_client.post('/model/rollback', json={'artifact': 'model_v1.pkl'})
        r = rollback_client.post('/predict', json=valid_payload)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 1

    def test_rollback_to_final_returns_200(self, rollback_client):
        r = rollback_client.post('/model/rollback', json={'artifact': 'final_model.pkl'})
        assert r.status_code == 200

    def test_sequential_rollback(self, rollback_client):
        r1 = rollback_client.post('/model/rollback', json={'artifact': 'model_v1.pkl'})
        r2 = rollback_client.post('/model/rollback', json={'artifact': 'final_model.pkl'})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()['active_model']['artifact'] == 'final_model.pkl'

    def test_rollback_actually_switches_active_model(self, rollback_client):
        import app.model_store as ms_mod
        info_initial = rollback_client.get('/model/info').json()
        assert info_initial['artifact'] == 'final_model.pkl'
        clf_initial = type(ms_mod.model_store.get_model().named_steps['classifier']).__name__
        r = rollback_client.post('/model/rollback', json={'artifact': 'model_v1.pkl'})
        assert r.status_code == 200
        clf_after = type(ms_mod.model_store.get_model().named_steps['classifier']).__name__
        assert clf_initial != clf_after, f'rollback did not switch the active model: still {clf_initial!r}'

class TestInvalidRollback:

    def test_nonexistent_artifact_returns_404(self, rollback_client):
        r = rollback_client.post('/model/rollback', json={'artifact': 'nonexistent_model.pkl'})
        assert r.status_code == 404

    def test_payload_without_artifact_returns_422(self, rollback_client):
        r = rollback_client.post('/model/rollback', json={})
        assert r.status_code == 422

    def test_corrupted_artifact_returns_400(self, rollback_client, tmp_path):
        import app.model_store as ms_mod
        bad_file = ms_mod.model_store._artifacts_dir / 'corrupted_model.pkl'
        bad_file.write_bytes(b'this is not a valid pickle')
        r = rollback_client.post('/model/rollback', json={'artifact': 'corrupted_model.pkl'})
        assert r.status_code == 400
