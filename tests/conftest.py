import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from crop_reco.config import ALL_CROPS, NUMERIC_FEATURES, TARGET

@pytest.fixture(scope='session')
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for crop in ALL_CROPS:
        for _ in range(6):
            rows.append({'N': float(rng.uniform(0, 140)), 'P': float(rng.uniform(5, 145)), 'K': float(rng.uniform(5, 205)), 'temperature': float(rng.uniform(8, 44)), 'humidity': float(rng.uniform(14, 99)), 'ph': float(rng.uniform(3.5, 9.5)), 'rainfall': float(rng.uniform(20, 300)), 'label': crop})
    return pd.DataFrame(rows)

@pytest.fixture(scope='session')
def dataset_df(small_df):
    return small_df

@pytest.fixture
def dataset_X(small_df):
    return small_df[NUMERIC_FEATURES].copy()

@pytest.fixture(scope='session')
def small_X(small_df):
    return small_df[NUMERIC_FEATURES].copy()

@pytest.fixture(scope='session')
def small_y(small_df):
    return small_df[TARGET].copy()

@pytest.fixture(scope='session')
def fitted_model(small_X, small_y):
    from sklearn.linear_model import LogisticRegression
    from crop_reco.modeling import build_full_pipeline
    model = build_full_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    model.fit(small_X, small_y)
    return model

@pytest.fixture(scope='session')
def api_artifacts_dir(fitted_model, tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp('api_arts')
    joblib.dump(fitted_model, tmp / 'final_model.pkl')
    return tmp

@pytest.fixture(scope='session')
def rollback_artifacts_dir(small_X, small_y, fitted_model, tmp_path_factory) -> Path:
    from sklearn.naive_bayes import GaussianNB
    from crop_reco.modeling import build_full_pipeline
    tmp = tmp_path_factory.mktemp('rollback_arts')
    joblib.dump(fitted_model, tmp / 'final_model.pkl')
    v1 = build_full_pipeline(GaussianNB())
    v1.fit(small_X, small_y)
    joblib.dump(v1, tmp / 'model_v1.pkl')
    return tmp

@pytest.fixture
def api_client(api_artifacts_dir):
    from fastapi.testclient import TestClient
    from app.main import app
    import app.model_store as ms_mod
    original_dir = ms_mod.model_store._artifacts_dir
    ms_mod.model_store._artifacts_dir = api_artifacts_dir
    with TestClient(app) as client:
        yield client
    ms_mod.model_store._artifacts_dir = original_dir

@pytest.fixture
def rollback_client(rollback_artifacts_dir):
    from fastapi.testclient import TestClient
    from app.main import app
    import app.model_store as ms_mod
    original_dir = ms_mod.model_store._artifacts_dir
    ms_mod.model_store._artifacts_dir = rollback_artifacts_dir
    with TestClient(app) as client:
        yield client
    ms_mod.model_store._artifacts_dir = original_dir

@pytest.fixture
def valid_record() -> dict:
    return {'N': 90.0, 'P': 42.0, 'K': 43.0, 'temperature': 20.9, 'humidity': 82.0, 'ph': 6.5, 'rainfall': 202.9}

@pytest.fixture
def valid_payload(valid_record) -> dict:
    return {'records': [valid_record]}
