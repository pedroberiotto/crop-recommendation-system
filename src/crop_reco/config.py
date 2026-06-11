from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
LOCAL_CSV = RAW_DATA_DIR / 'Crop_recommendation.csv'
KAGGLE_DATASET = 'atharvaingle/crop-recommendation-dataset'
KAGGLE_FILE = 'Crop_recommendation.csv'
import os as _os
ARTIFACTS_DIR = Path(_os.environ['ARTIFACTS_DIR']) if 'ARTIFACTS_DIR' in _os.environ else PROJECT_ROOT / 'artifacts'
PIPELINE_PATH = ARTIFACTS_DIR / 'preprocessing_pipeline.pkl'
MODEL_PATH = ARTIFACTS_DIR / 'final_model.pkl'
MODEL_FINAL_PATH = MODEL_PATH
MODEL_META_PATH = ARTIFACTS_DIR / 'final_model_meta.json'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
MLFLOW_DIR = PROJECT_ROOT / 'mlflow'
MLFLOW_DB_PATH = MLFLOW_DIR / 'mlflow.db'
MLFLOW_ARTIFACTS_PATH = MLFLOW_DIR / 'artifacts'
MLFLOW_TRACKING_URI_DEFAULT = f'sqlite:///{MLFLOW_DB_PATH}'
MLFLOW_EXPERIMENT_NAME = 'crop-recommendation'
MLFLOW_MODEL_NAME = 'crop-recommender'
NUMERIC_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'
ALL_COLUMNS = NUMERIC_FEATURES + [TARGET]
ENGINEERED_FEATURES = ['NP_ratio', 'NK_ratio', 'PK_ratio', 'NPK_sum']
FINAL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES
FEATURE_BOUNDS = {'N': (0.0, 300.0), 'P': (0.0, 300.0), 'K': (0.0, 300.0), 'temperature': (-20.0, 60.0), 'humidity': (0.0, 100.0), 'ph': (0.0, 14.0), 'rainfall': (0.0, 1000.0)}
ALL_CROPS = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
import os as _os
LOG_FILE = _os.environ.get('LOG_FILE', '')
