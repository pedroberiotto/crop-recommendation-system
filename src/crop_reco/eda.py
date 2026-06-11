from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('.matplotlib-cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .config import FEATURE_BOUNDS, FIGURES_DIR, NUMERIC_FEATURES, TARGET

def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _save(fig, path, dpi=120):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return Path(path)

def plot_class_distribution(df, save_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    counts = df[TARGET].value_counts().sort_values()
    counts.plot.barh(ax=ax, color='#4c72b0', edgecolor='white')
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=9)
    ax.set_xlabel('Count')
    ax.set_ylabel('Crop')
    ax.set_title(f'Distribution of {df[TARGET].nunique()} classes (total = {len(df)})')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    return _save(fig, save_path)

def plot_feature_distributions(df, save_path):
    n = len(NUMERIC_FEATURES)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes = axes.flatten()
    for ax, feat in zip(axes, NUMERIC_FEATURES):
        ax.hist(df[feat], bins=40, color='#4c72b0', alpha=0.85, edgecolor='white')
        ax.axvline(df[feat].mean(), color='#dd8452', linestyle='--', linewidth=1.5, label=f'mean = {df[feat].mean():.1f}')
        ax.set_title(feat)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle('Global feature distribution', fontsize=14, y=1.01)
    return _save(fig, save_path)

def plot_boxplots_by_class(df, save_path, features=None):
    features = features if features is not None else NUMERIC_FEATURES
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]
    order = sorted(df[TARGET].unique())
    positions = np.arange(1, len(order) + 1)
    for ax, feat in zip(axes, features):
        values = [df.loc[df[TARGET] == label, feat].to_numpy() for label in order]
        ax.boxplot(values, positions=positions, patch_artist=True, showfliers=True)
        ax.set_title(f'{feat} by crop')
        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=75, ha='right')
        ax.set_xlabel('')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    return _save(fig, save_path)

def plot_violin_by_class(df, save_path, features=None):
    features = features if features is not None else ['N', 'P', 'K', 'ph']
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]
    order = sorted(df[TARGET].unique())
    positions = np.arange(1, len(order) + 1)
    for ax, feat in zip(axes, features):
        values = [df.loc[df[TARGET] == label, feat].to_numpy() for label in order]
        ax.violinplot(values, positions=positions, showmeans=False, showmedians=True, showextrema=True)
        ax.set_title(f'{feat} by crop (density)')
        ax.set_xticks(positions)
        ax.set_xticklabels(order, rotation=75, ha='right')
        ax.set_xlabel('')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    return _save(fig, save_path)

def plot_correlation_heatmap(df, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[NUMERIC_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'shrink': 0.75})
    ax.set_title('Pearson correlation matrix — soil/climate features')
    return _save(fig, save_path)

def plot_class_signature(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    means = df.groupby(TARGET)[NUMERIC_FEATURES].mean()
    z = (means - means.mean()) / means.std()
    sns.heatmap(z, cmap='RdBu_r', center=0, annot=True, fmt='.1f', cbar_kws={'label': 'z-score (across crops)'}, ax=ax, linewidths=0.3, linecolor='white')
    ax.set_title('Mean nutritional/climatic signature per crop\n(z-score across 22 crops)')
    ax.set_xlabel('')
    ax.set_ylabel('Crop')
    return _save(fig, save_path)

def plot_pairplot_sample(df, save_path, classes=None, max_per_class=30, random_state=42):
    classes = classes if classes is not None else ['rice', 'coffee', 'grapes', 'watermelon']
    sub = df[df[TARGET].isin(classes)].copy()
    if max_per_class is not None:
        sub = pd.concat([g.sample(n=min(len(g), max_per_class), random_state=random_state) for _, g in sub.groupby(TARGET, sort=False)], ignore_index=True)
    features = NUMERIC_FEATURES
    labels = list(sub[TARGET].drop_duplicates())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
    color_by_label = dict(zip(labels, colors))
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(1.45 * n, 1.45 * n))
    for i, y_feat in enumerate(features):
        for j, x_feat in enumerate(features):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                for label in labels:
                    values = sub.loc[sub[TARGET] == label, x_feat]
                    ax.hist(values, bins=12, alpha=0.45, color=color_by_label[label])
            else:
                for label in labels:
                    part = sub[sub[TARGET] == label]
                    ax.scatter(part[x_feat], part[y_feat], s=10, alpha=0.65, color=color_by_label[label], edgecolors='none')
            if i == n - 1:
                ax.set_xlabel(x_feat, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(y_feat, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis='both', labelsize=6)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=color_by_label[label], label=label, markersize=5) for label in labels]
    fig.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)
    fig.suptitle(f"Pairplot — sampled crops: {', '.join(classes)}", y=1.01, fontsize=12)
    return _save(fig, save_path, dpi=110)

def summarize_outliers_iqr(df):
    rows = []
    for col in NUMERIC_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        mask = (df[col] < lo) | (df[col] > hi)
        rows.append({'feature': col, 'n_outliers': int(mask.sum()), 'pct_outliers': float(round(100 * mask.mean(), 2)), 'lower_bound': float(round(lo, 2)), 'upper_bound': float(round(hi, 2)), 'min': float(round(df[col].min(), 2)), 'max': float(round(df[col].max(), 2))})
    return pd.DataFrame(rows)

def physical_validity_report(df):
    rows = []
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col not in df.columns:
            continue
        below = int((df[col] < lo).sum())
        above = int((df[col] > hi).sum())
        rows.append({'feature': col, 'lower_physical': lo, 'upper_physical': hi, 'n_below': below, 'n_above': above, 'n_total_violations': below + above})
    return pd.DataFrame(rows)

def class_statistics(df):
    stats = df.groupby(TARGET)[NUMERIC_FEATURES].agg(['mean', 'std', 'min', 'max'])
    return stats.round(2)

def run_eda(df, output_dir=None):
    output_dir = _ensure_dir(output_dir if output_dir is not None else FIGURES_DIR)
    paths = {'class_distribution': plot_class_distribution(df, output_dir / '01_class_distribution.png'), 'feature_distributions': plot_feature_distributions(df, output_dir / '02_feature_distributions.png'), 'boxplots_by_class': plot_boxplots_by_class(df, output_dir / '03_boxplots_by_class.png'), 'violins_by_class': plot_violin_by_class(df, output_dir / '04_violins_by_class.png'), 'correlation_heatmap': plot_correlation_heatmap(df, output_dir / '05_correlation_heatmap.png'), 'class_signature': plot_class_signature(df, output_dir / '06_class_signature.png'), 'pairplot': plot_pairplot_sample(df, output_dir / '07_pairplot_sample.png')}
    return paths
