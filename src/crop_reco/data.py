from pathlib import Path
import pandas as pd
from .config import ALL_COLUMNS, KAGGLE_DATASET, KAGGLE_FILE, LOCAL_CSV, NUMERIC_FEATURES, TARGET

def download_dataset(dest: Path=LOCAL_CSV) -> Path:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, KAGGLE_DATASET, KAGGLE_FILE)
    except Exception as exc:
        raise RuntimeError(f"Failed to download '{KAGGLE_DATASET}' from Kaggle.") from exc
    validate_schema(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest

def load_dataset() -> pd.DataFrame:
    if LOCAL_CSV.exists():
        return load_local(LOCAL_CSV)
    download_dataset(LOCAL_CSV)
    return load_local(LOCAL_CSV)

def load_local(path: Path=LOCAL_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    df = pd.read_csv(path)
    validate_schema(df)
    return df

def validate_schema(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Expected pd.DataFrame, got {type(df).__name__}')
    cols = set(df.columns)
    expected = set(ALL_COLUMNS)
    missing = expected - cols
    extra = cols - expected
    if missing or extra:
        raise ValueError(f'Invalid schema. Missing: {sorted(missing)}. Extra: {sorted(extra)}.')

def split_xy(df: pd.DataFrame):
    validate_schema(df)
    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET].copy()
    return (X, y)
