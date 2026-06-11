import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import ENGINEERED_FEATURES, FEATURE_BOUNDS, FINAL_FEATURES, NUMERIC_FEATURES

class FeatureValidator(BaseEstimator, TransformerMixin):

    def __init__(self, expected_columns=None):
        self.expected_columns = expected_columns if expected_columns is not None else NUMERIC_FEATURES

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def transform(self, X):
        self._validate(X)
        return X[self.expected_columns].copy()

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'Expected pd.DataFrame, got {type(X).__name__}')
        missing = set(self.expected_columns) - set(X.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        nan_cols = [c for c in self.expected_columns if X[c].isna().any()]
        if nan_cols:
            raise ValueError(f'NaN detected in: {nan_cols}')
        non_numeric = [c for c in self.expected_columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            raise ValueError(f'Non-numeric columns: {non_numeric}')

    def get_feature_names_out(self, input_features=None):
        return list(self.expected_columns)

class PhysicalBoundsClipper(BaseEstimator, TransformerMixin):

    def __init__(self, bounds=None):
        self.bounds = bounds if bounds is not None else FEATURE_BOUNDS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return list(input_features)
        return list(NUMERIC_FEATURES)

class NutrientRatioEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, epsilon=0.001):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['NP_ratio'] = X['N'] / (X['P'] + self.epsilon)
        X['NK_ratio'] = X['N'] / (X['K'] + self.epsilon)
        X['PK_ratio'] = X['P'] / (X['K'] + self.epsilon)
        X['NPK_sum'] = X['N'] + X['P'] + X['K']
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else NUMERIC_FEATURES
        return base + ENGINEERED_FEATURES

def build_pipeline():
    scaler_step = ColumnTransformer(transformers=[('standard', StandardScaler(), FINAL_FEATURES)], remainder='drop', verbose_feature_names_out=False)
    return Pipeline(steps=[('validator', FeatureValidator(expected_columns=NUMERIC_FEATURES)), ('clipper', PhysicalBoundsClipper()), ('engineer', NutrientRatioEngineer()), ('scaler', scaler_step)])
