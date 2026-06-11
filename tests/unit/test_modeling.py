from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from crop_reco.config import NUMERIC_FEATURES, TARGET
from crop_reco.modeling import build_full_pipeline, evaluate_model, get_candidate_models, run_cross_validation, train_final_model

class TestGetCandidateModels:

    def test_returns_four_models(self):
        models = get_candidate_models()
        assert len(models) == 4

    def test_correct_names(self):
        expected = {'logistic_regression', 'gaussian_nb', 'random_forest', 'gradient_boosting'}
        assert set(get_candidate_models().keys()) == expected

    def test_all_have_predict_proba(self):
        for name, clf in get_candidate_models().items():
            assert hasattr(clf, 'predict_proba'), f'{name} without predict_proba'

    def test_knn_excluded_from_production(self):
        assert 'knn' not in get_candidate_models()

class TestBuildFullPipeline:

    def test_returns_sklearn_pipeline(self):
        p = build_full_pipeline(LogisticRegression())
        assert isinstance(p, Pipeline)

    def test_required_steps(self):
        p = build_full_pipeline(LogisticRegression())
        assert 'preprocessor' in p.named_steps
        assert 'classifier' in p.named_steps

    def test_preprocessor_is_new_instance(self):
        p1 = build_full_pipeline(LogisticRegression())
        p2 = build_full_pipeline(LogisticRegression())
        assert p1.named_steps['preprocessor'] is not p2.named_steps['preprocessor']

    def test_fit_predict_proba_shape(self, small_X, small_y):
        model = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
        model.fit(small_X, small_y)
        proba = model.predict_proba(small_X)
        assert proba.shape == (len(small_X), 22)

    def test_predict_proba_sums_to_one(self, small_X, small_y):
        model = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
        model.fit(small_X, small_y)
        proba = model.predict_proba(small_X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-06)

    def test_all_22_classes_present(self, small_X, small_y):
        model = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
        model.fit(small_X, small_y)
        assert len(model.classes_) == 22

    def test_predict_proba_not_negative(self, small_X, small_y):
        model = build_full_pipeline(LogisticRegression(max_iter=500, random_state=42))
        model.fit(small_X, small_y)
        proba = model.predict_proba(small_X)
        assert (proba >= 0).all()

class TestRunCrossValidation:

    def test_returns_dataframe(self, small_X, small_y):
        results = run_cross_validation(small_X, small_y, n_splits=2)
        assert isinstance(results, pd.DataFrame)

    def test_required_columns(self, small_X, small_y):
        results = run_cross_validation(small_X, small_y, n_splits=2)
        required = {'model', 'accuracy_mean', 'accuracy_std', 'f1_macro_mean', 'f1_macro_std'}
        assert required.issubset(set(results.columns))

    def test_number_of_rows(self, small_X, small_y):
        results = run_cross_validation(small_X, small_y, n_splits=2)
        assert len(results) == 4

    def test_sorted_by_f1_descending(self, small_X, small_y):
        results = run_cross_validation(small_X, small_y, n_splits=2)
        f1_vals = results['f1_macro_mean'].tolist()
        assert f1_vals == sorted(f1_vals, reverse=True)

    def test_metrics_between_zero_and_one(self, small_X, small_y):
        results = run_cross_validation(small_X, small_y, n_splits=2)
        for col in ['accuracy_mean', 'f1_macro_mean']:
            assert results[col].between(0.0, 1.0).all(), f'{col} out of [0,1]'

    def test_std_not_negative(self, small_X, small_y):
        results = run_cross_validation(small_X, small_y, n_splits=2)
        for col in ['accuracy_std', 'f1_macro_std']:
            assert (results[col] >= 0).all()

class TestTrainFinalModel:

    def test_returns_fitted_pipeline(self, small_X, small_y):
        clf = LogisticRegression(max_iter=500, random_state=42)
        model = train_final_model(small_X, small_y, clf)
        assert isinstance(model, Pipeline)
        assert hasattr(model, 'classes_')

    def test_predicts_without_error(self, small_X, small_y):
        clf = LogisticRegression(max_iter=500, random_state=42)
        model = train_final_model(small_X, small_y, clf)
        preds = model.predict(small_X)
        assert len(preds) == len(small_X)

    def test_preprocessor_fitted_independently(self, small_X, small_y):
        clf1 = LogisticRegression(max_iter=500, random_state=42)
        clf2 = LogisticRegression(max_iter=500, random_state=42)
        m1 = train_final_model(small_X, small_y, clf1)
        m2 = train_final_model(small_X, small_y, clf2)
        assert m1.named_steps['preprocessor'] is not m2.named_steps['preprocessor']

class TestEvaluateModel:

    def test_returns_accuracy_and_f1(self, fitted_model, small_X, small_y):
        metrics = evaluate_model(fitted_model, small_X, small_y)
        assert 'accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_macro'] <= 1.0

    def test_returns_report_dict_and_str(self, fitted_model, small_X, small_y):
        metrics = evaluate_model(fitted_model, small_X, small_y)
        assert isinstance(metrics['report_dict'], dict)
        assert isinstance(metrics['report_str'], str)
        assert 'precision' in metrics['report_str']

    def test_saves_figures(self, fitted_model, small_X, small_y, tmp_path):
        metrics = evaluate_model(fitted_model, small_X, small_y, figures_dir=tmp_path)
        assert metrics['paths']['confusion_matrix'].exists()
        assert metrics['paths']['classification_report'].exists()
        assert metrics['paths']['confusion_matrix'].stat().st_size > 0

    def test_does_not_save_without_figures_dir(self, fitted_model, small_X, small_y):
        metrics = evaluate_model(fitted_model, small_X, small_y, figures_dir=None)
        assert metrics['paths'] == {}
