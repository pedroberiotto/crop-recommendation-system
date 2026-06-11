import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from crop_reco.config import ENGINEERED_FEATURES, FINAL_FEATURES, NUMERIC_FEATURES
from crop_reco.preprocessing import FeatureValidator, NutrientRatioEngineer, PhysicalBoundsClipper, build_pipeline

class TestFeatureValidator:

    def test_accepts_valid_df(self, dataset_X):
        out = FeatureValidator().fit_transform(dataset_X)
        assert list(out.columns) == NUMERIC_FEATURES

    def test_reorders_columns(self, dataset_X):
        shuffled = dataset_X[list(reversed(NUMERIC_FEATURES))]
        out = FeatureValidator().fit_transform(shuffled)
        assert list(out.columns) == NUMERIC_FEATURES

    def test_fails_without_column(self, dataset_X):
        with pytest.raises(ValueError, match='Missing'):
            FeatureValidator().fit_transform(dataset_X.drop(columns=['ph']))

    def test_fails_with_nan(self, dataset_X):
        X = dataset_X.copy()
        X.loc[0, 'N'] = np.nan
        with pytest.raises(ValueError, match='NaN'):
            FeatureValidator().fit_transform(X)

    def test_fails_if_not_dataframe(self):
        with pytest.raises(TypeError):
            FeatureValidator().fit_transform(np.zeros((5, 7)))

    def test_fails_with_non_numeric_column(self, dataset_X):
        X = dataset_X.copy()
        X['N'] = X['N'].astype(str)
        with pytest.raises(ValueError, match='Non-numeric'):
            FeatureValidator().fit_transform(X)

class TestPhysicalBoundsClipper:

    def test_does_not_alter_in_range_data(self, dataset_X):
        out = PhysicalBoundsClipper().fit_transform(dataset_X)
        pd.testing.assert_frame_equal(out, dataset_X)

    def test_clips_above_upper_bound(self, dataset_X):
        X = dataset_X.copy()
        X.loc[0, 'ph'] = 25.0
        out = PhysicalBoundsClipper().fit_transform(X)
        assert out.loc[0, 'ph'] == 14.0

    def test_clips_below_lower_bound(self, dataset_X):
        X = dataset_X.copy()
        X.loc[0, 'N'] = -50.0
        out = PhysicalBoundsClipper().fit_transform(X)
        assert out.loc[0, 'N'] == 0.0

class TestNutrientRatioEngineer:

    def test_adds_features(self, dataset_X):
        out = NutrientRatioEngineer().fit_transform(dataset_X)
        for f in ENGINEERED_FEATURES:
            assert f in out.columns
        assert out.shape == (len(dataset_X), len(NUMERIC_FEATURES) + len(ENGINEERED_FEATURES))

    def test_ratios_computed_correctly(self, dataset_X):
        out = NutrientRatioEngineer(epsilon=0.001).fit_transform(dataset_X)
        np.testing.assert_allclose(out['NP_ratio'], dataset_X['N'] / (dataset_X['P'] + 0.001))
        np.testing.assert_allclose(out['NPK_sum'], dataset_X['N'] + dataset_X['P'] + dataset_X['K'])

    def test_robust_to_zero_p(self):
        X = pd.DataFrame({'N': [10.0], 'P': [0.0], 'K': [10.0], 'temperature': [25.0], 'humidity': [60.0], 'ph': [7.0], 'rainfall': [100.0]})
        out = NutrientRatioEngineer().fit_transform(X)
        assert np.isfinite(out['NP_ratio']).all()

class TestBuildPipeline:

    def test_returns_sklearn_pipeline(self):
        p = build_pipeline()
        assert isinstance(p, Pipeline)
        for step in ['validator', 'clipper', 'engineer', 'scaler']:
            assert step in p.named_steps

    def test_output_shape(self, dataset_X):
        out = build_pipeline().fit_transform(dataset_X)
        assert out.shape == (len(dataset_X), len(FINAL_FEATURES))

    def test_standardization(self, dataset_X):
        out = build_pipeline().fit_transform(dataset_X)
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(out.std(axis=0), 1, atol=0.01)

    def test_consistent_transform(self, dataset_X):
        p = build_pipeline()
        a = p.fit_transform(dataset_X)
        b = p.transform(dataset_X)
        np.testing.assert_allclose(a, b)

    def test_fit_idempotence(self, dataset_X):
        p1 = build_pipeline().fit(dataset_X)
        p2 = build_pipeline().fit(dataset_X)
        s1 = p1.named_steps['scaler'].named_transformers_['standard']
        s2 = p2.named_steps['scaler'].named_transformers_['standard']
        np.testing.assert_allclose(s1.mean_, s2.mean_)

class TestSerialization:

    def test_pickle_round_trip(self, dataset_X, tmp_path):
        p = build_pipeline().fit(dataset_X)
        expected = p.transform(dataset_X)
        path = tmp_path / 'p.pkl'
        joblib.dump(p, path)
        reloaded = joblib.load(path)
        np.testing.assert_allclose(expected, reloaded.transform(dataset_X))

    def test_applicable_to_new_data(self, dataset_X):
        p = build_pipeline().fit(dataset_X)
        out = p.transform(dataset_X.head(5))
        assert out.shape == (5, len(FINAL_FEATURES))

class TestNoDataLeakage:

    def test_scaler_learns_only_on_train(self, dataset_X):
        train = dataset_X.iloc[:50]
        test = dataset_X.iloc[50:]
        p = build_pipeline().fit(train)
        s = p.named_steps['scaler'].named_transformers_['standard']
        clipped_tr = p.named_steps['clipper'].transform(p.named_steps['validator'].transform(train))
        engineered_tr = p.named_steps['engineer'].transform(clipped_tr)
        np.testing.assert_allclose(s.mean_, engineered_tr.mean(axis=0).values, atol=1e-10)
        clipped_te = p.named_steps['clipper'].transform(p.named_steps['validator'].transform(test))
        engineered_te = p.named_steps['engineer'].transform(clipped_te)
        diff = np.abs(s.mean_ - engineered_te.mean(axis=0).values).sum()
        assert diff > 0
