import pytest
from crop_reco.config import LOCAL_CSV, NUMERIC_FEATURES, TARGET
from crop_reco.data import load_local, split_xy, validate_schema

@pytest.mark.skipif(not LOCAL_CSV.exists(), reason='Real CSV not found — run `make download` to download from Kaggle')
class TestRealDataset:

    def test_shape_and_balance(self):
        df = load_local(LOCAL_CSV)
        assert df.shape == (2200, 8)
        assert df['label'].nunique() == 22
        assert df['label'].value_counts().nunique() == 1
        assert df['label'].value_counts().iloc[0] == 100

class TestValidateSchema:

    def test_accepts_correct_schema(self, dataset_df):
        validate_schema(dataset_df)

    def test_fails_if_missing_column(self, dataset_df):
        with pytest.raises(ValueError, match='Missing'):
            validate_schema(dataset_df.drop(columns=['N']))

    def test_fails_if_extra_column(self, dataset_df):
        with pytest.raises(ValueError, match='Extra'):
            validate_schema(dataset_df.assign(extra=1))

    def test_fails_if_not_dataframe(self):
        with pytest.raises(TypeError):
            validate_schema([1, 2, 3])

class TestLoadLocal:

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_local(tmp_path / 'does_not_exist.csv')

    def test_loads_valid_csv(self, tmp_path, dataset_df):
        csv = tmp_path / 'sample.csv'
        dataset_df.to_csv(csv, index=False)
        df = load_local(csv)
        assert len(df) == len(dataset_df)
        assert set(df.columns) == set(dataset_df.columns)

class TestSplitXY:

    def test_correct_separation(self, dataset_df):
        X, y = split_xy(dataset_df)
        assert list(X.columns) == NUMERIC_FEATURES
        assert y.name == TARGET
        assert len(X) == len(y) == len(dataset_df)

    def test_x_does_not_contain_target(self, dataset_df):
        X, _ = split_xy(dataset_df)
        assert TARGET not in X.columns
