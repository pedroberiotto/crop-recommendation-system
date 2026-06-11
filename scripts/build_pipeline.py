import sys
from pathlib import Path
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crop_reco.config import FIGURES_DIR, PIPELINE_PATH
from crop_reco.data import load_dataset, split_xy
from crop_reco.eda import run_eda
from crop_reco.preprocessing import build_pipeline

def main():
    print('==> 1/4  Loading dataset')
    df = load_dataset()
    print(f"        shape = {df.shape} | classes = {df['label'].nunique()}")
    print('==> 2/4  Generating exploratory analysis (charts)')
    paths = run_eda(df, FIGURES_DIR)
    for name, p in paths.items():
        print(f'        {name:25s} -> {p.relative_to(p.parents[2])}')
    print('==> 3/4  Training preprocessing pipeline')
    X, y = split_xy(df)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    print(f'        steps: {[s for s, _ in pipeline.steps]}')
    print(f'==> 4/4  Serializing pipeline to {PIPELINE_PATH}')
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    reloaded = joblib.load(PIPELINE_PATH)
    sample = X.head(3)
    out = reloaded.transform(sample)
    print(f'        sanity check: input {sample.shape} -> output {out.shape}')
    print('\nOK — pipeline ready for use by the inference API.')
if __name__ == '__main__':
    main()
