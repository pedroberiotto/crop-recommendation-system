from __future__ import annotations
import pytest

class TestPredictSingle:

    def test_returns_200(self, api_client, valid_payload):
        r = api_client.post('/predict', json=valid_payload)
        assert r.status_code == 200

    def test_response_structure(self, api_client, valid_payload):
        r = api_client.post('/predict', json=valid_payload)
        body = r.json()
        assert 'predictions' in body
        assert len(body['predictions']) == 1
        pred = body['predictions'][0]
        assert 'crop' in pred
        assert 'confidence' in pred
        assert 'alternatives' in pred
        assert len(pred['alternatives']) == 3

    def test_confidence_between_0_and_1(self, api_client, valid_payload):
        r = api_client.post('/predict', json=valid_payload)
        pred = r.json()['predictions'][0]
        assert 0.0 <= pred['confidence'] <= 1.0

    def test_alternatives_have_crop_and_probability(self, api_client, valid_payload):
        r = api_client.post('/predict', json=valid_payload)
        alts = r.json()['predictions'][0]['alternatives']
        for alt in alts:
            assert 'crop' in alt
            assert 'probability' in alt
            assert 0.0 <= alt['probability'] <= 1.0

    def test_crop_is_non_empty_string(self, api_client, valid_payload):
        r = api_client.post('/predict', json=valid_payload)
        crop = r.json()['predictions'][0]['crop']
        assert isinstance(crop, str) and len(crop) > 0

class TestPredictBatch:

    def test_batch_of_5_returns_5_predictions(self, api_client, valid_record):
        payload = {'records': [valid_record] * 5}
        r = api_client.post('/predict', json=payload)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 5

    def test_batch_of_50_returns_200(self, api_client, valid_record):
        payload = {'records': [valid_record] * 50}
        r = api_client.post('/predict', json=payload)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 50

    def test_max_batch_500(self, api_client, valid_record):
        payload = {'records': [valid_record] * 500}
        r = api_client.post('/predict', json=payload)
        assert r.status_code == 200
        assert len(r.json()['predictions']) == 500

    def test_batch_above_500_returns_422(self, api_client, valid_record):
        payload = {'records': [valid_record] * 501}
        r = api_client.post('/predict', json=payload)
        assert r.status_code == 422

class TestPayloadValidation:

    def test_required_field_missing_returns_422(self, api_client, valid_record):
        record = {k: v for k, v in valid_record.items() if k != 'ph'}
        r = api_client.post('/predict', json={'records': [record]})
        assert r.status_code == 422

    def test_empty_list_returns_422(self, api_client):
        r = api_client.post('/predict', json={'records': []})
        assert r.status_code == 422

    def test_empty_body_returns_422(self, api_client):
        r = api_client.post('/predict', json={})
        assert r.status_code == 422

    def test_negative_ph_returns_422(self, api_client, valid_record):
        record = {**valid_record, 'ph': -1.0}
        r = api_client.post('/predict', json={'records': [record]})
        assert r.status_code == 422

    def test_humidity_above_100_returns_422(self, api_client, valid_record):
        record = {**valid_record, 'humidity': 150.0}
        r = api_client.post('/predict', json={'records': [record]})
        assert r.status_code == 422

    def test_negative_n_returns_422(self, api_client, valid_record):
        record = {**valid_record, 'N': -10.0}
        r = api_client.post('/predict', json={'records': [record]})
        assert r.status_code == 422

    def test_wrong_type_returns_422(self, api_client, valid_record):
        record = {**valid_record, 'N': 'high'}
        r = api_client.post('/predict', json={'records': [record]})
        assert r.status_code == 422

class TestExplainQueryParam:

    def test_explain_false_does_not_return_explanation(self, api_client, valid_payload):
        r = api_client.post('/predict?explain=false', json=valid_payload)
        assert r.status_code == 200
        pred = r.json()['predictions'][0]
        assert pred.get('explanation') is None

    def test_explain_true_field_present_or_null(self, api_client, valid_payload):
        r = api_client.post('/predict?explain=true', json=valid_payload)
        assert r.status_code == 200
        pred = r.json()['predictions'][0]
        assert 'explanation' in pred or pred.get('confidence', 0) <= 0.7

class TestStrictQueryParam:

    def test_strict_false_accepts_invalid_ph(self, api_client, valid_record):
        record = {**valid_record, 'ph': -1.0}
        r = api_client.post('/predict?strict=false', json={'records': [record]})
        assert r.status_code == 200, r.text

    def test_strict_false_accepts_humidity_above_100(self, api_client, valid_record):
        record = {**valid_record, 'humidity': 150.0}
        r = api_client.post('/predict?strict=false', json={'records': [record]})
        assert r.status_code == 200

    def test_strict_default_is_true(self, api_client, valid_record):
        record = {**valid_record, 'ph': -1.0}
        r = api_client.post('/predict', json={'records': [record]})
        assert r.status_code == 422

    def test_strict_false_still_rejects_missing_field(self, api_client, valid_record):
        record = {k: v for k, v in valid_record.items() if k != 'ph'}
        r = api_client.post('/predict?strict=false', json={'records': [record]})
        assert r.status_code == 422
