"""Harmonize five annual CSVs and report descriptive associations."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FILES = [Path(f'{year}.csv') for year in range(2015, 2020)]
FEATURES = ['GDP per capita', 'Social support', 'Healthy life expectancy',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
COLUMN_MAP = {
    'Country or region': 'Country', 'Happiness.Score': 'Happiness Score',
    'Score': 'Happiness Score', 'Economy (GDP per Capita)': 'GDP per capita',
    'Economy..GDP.per.Capita.': 'GDP per capita', 'Family': 'Social support',
    'Health (Life Expectancy)': 'Healthy life expectancy',
    'Health..Life.Expectancy.': 'Healthy life expectancy',
    'Freedom': 'Freedom to make life choices',
    'Trust (Government Corruption)': 'Perceptions of corruption',
    'Trust..Government.Corruption.': 'Perceptions of corruption',
}


def load_and_harmonize(files=FILES):
    annual = []
    hashes = {}
    for path in files:
        path = Path(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path).rename(columns=COLUMN_MAP)
        frame['Year'] = int(path.stem)
        if 'Region' not in frame:
            frame['Region'] = pd.NA
        annual.append(frame)
    merged = pd.concat(annual, ignore_index=True)
    columns = ['Country', 'Region', 'Year', 'Happiness Score', *FEATURES]
    merged = merged[columns]
    # The 2018/2019 files omit region; use a unique earlier mapping where known.
    known = merged.dropna(subset=['Region']).groupby('Country')['Region'].agg(lambda x: x.mode().iloc[0])
    merged['Region'] = merged['Region'].fillna(merged['Country'].map(known))
    for column in ['Happiness Score', *FEATURES]:
        merged[column] = pd.to_numeric(merged[column], errors='coerce')
    return merged, hashes


def grouped_association(merged):
    clean = merged.dropna(subset=['Happiness Score', *FEATURES]).copy()
    if clean['Country'].nunique() < 5:
        raise ValueError('Need at least five countries for a country-held-out check')
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(clean[FEATURES], groups=clean['Country']))
    train, test = clean.iloc[train_idx], clean.iloc[test_idx]
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train[FEATURES], train['Happiness Score'])
    prediction = model.predict(test[FEATURES])
    coefficients = pd.Series(model[-1].coef_, index=FEATURES).sort_values()
    return clean, coefficients, {
        'country_held_out_r2': float(r2_score(test['Happiness Score'], prediction)),
        'country_held_out_mae': float(mean_absolute_error(test['Happiness Score'], prediction)),
        'train_countries': int(train['Country'].nunique()),
        'test_countries': int(test['Country'].nunique()),
    }


def main():
    merged, hashes = load_and_harmonize()
    merged.to_csv('world_happiness_report.csv', index=False)
    clean, coefficients, metrics = grouped_association(merged)
    out = Path('vizualization')
    out.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=coefficients.values, y=coefficients.index)
    plt.title('Standardized associations with Happiness Score')
    plt.tight_layout()
    plt.savefig(out / 'standardized_associations.png')
    plt.close()
    regional = clean.dropna(subset=['Region']).groupby('Region')['Happiness Score'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    regional.plot(kind='barh')
    plt.title('Mean score by mapped region (available rows)')
    plt.tight_layout()
    plt.savefig(out / 'happiness_by_region.png')
    plt.close()
    corr = clean[['Happiness Score', *FEATURES]].corr()
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Descriptive correlations')
    plt.tight_layout()
    plt.savefig(out / 'correlation_heatmap.png')
    plt.close()
    report = {'input_sha256': hashes, 'metrics': metrics,
              'region_rows_mapped': int(merged['Region'].notna().sum()),
              'region_rows_total': len(merged),
              'limits': 'Associations are not causal; CSV publisher lineage and redistribution rights are unverified.'}
    (out / 'validation_results.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
