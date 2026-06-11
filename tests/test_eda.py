import matplotlib
matplotlib.use('Agg')
import pandas as pd
from crop_reco.eda import class_statistics, physical_validity_report, run_eda, summarize_outliers_iqr

class TestReports:

    def test_summarize_outliers_iqr(self, dataset_df):
        out = summarize_outliers_iqr(dataset_df)
        assert len(out) == 7
        assert {'feature', 'n_outliers', 'pct_outliers', 'lower_bound', 'upper_bound'}.issubset(out.columns)

    def test_physical_validity_no_sample(self, dataset_df):
        out = physical_validity_report(dataset_df)
        assert (out['n_total_violations'] == 0).all()

    def test_class_statistics_shape(self, dataset_df):
        stats = class_statistics(dataset_df)
        assert len(stats) == dataset_df['label'].nunique()

class TestRunEDA:

    def test_generates_all_charts(self, dataset_df, tmp_path):
        paths = run_eda(dataset_df, output_dir=tmp_path)
        expected = {'class_distribution', 'feature_distributions', 'boxplots_by_class', 'violins_by_class', 'correlation_heatmap', 'class_signature', 'pairplot'}
        assert set(paths.keys()) == expected
        for p in paths.values():
            assert p.exists()
            assert p.stat().st_size > 0
