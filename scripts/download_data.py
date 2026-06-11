import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crop_reco.config import LOCAL_CSV
from crop_reco.data import download_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description='Download Crop Recommendation dataset')
    parser.add_argument('--force', action='store_true', help='Overwrite local CSV even if it already exists')
    args = parser.parse_args()
    if LOCAL_CSV.exists() and (not args.force):
        print(f'✓ Dataset already present at {LOCAL_CSV} ({LOCAL_CSV.stat().st_size:,} bytes).')
        print('  Use --force to re-download.')
        return
    print(f"⬇  Downloading '{LOCAL_CSV.name}' from Kaggle…")
    dest = download_dataset(LOCAL_CSV)
    import pandas as pd
    df = __import__('pandas').read_csv(dest)
    print(f'✓  Saved to {dest}')
    print(f"   shape={df.shape} | classes={df['label'].nunique()}")
if __name__ == '__main__':
    main()
