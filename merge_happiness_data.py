import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Merge CSV files
files = [
    '2015.csv', '2016.csv', '2017.csv', '2018.csv', '2019.csv'
]

col_map = {
    # 2015/2016
    'Country': 'Country',
    'Country or region': 'Country',
    'Region': 'Region',
    'Happiness Score': 'Happiness Score',
    'Score': 'Happiness Score',
    'Happiness.Score': 'Happiness Score',
    'Happiness Rank': 'Happiness Rank',
    'Happiness.Rank': 'Happiness Rank',
    'Overall rank': 'Happiness Rank',
    'Economy (GDP per Capita)': 'GDP per capita',
    'Economy..GDP.per.Capita.': 'GDP per capita',
    'GDP per capita': 'GDP per capita',
    'Social support': 'Social support',
    'Family': 'Social support',
    'Health (Life Expectancy)': 'Healthy life expectancy',
    'Health..Life.Expectancy.': 'Healthy life expectancy',
    'Healthy life expectancy': 'Healthy life expectancy',
    'Freedom': 'Freedom to make life choices',
    'Freedom to make life choices': 'Freedom to make life choices',
    'Generosity': 'Generosity',
    'Trust (Government Corruption)': 'Perceptions of corruption',
    'Trust..Government.Corruption.': 'Perceptions of corruption',
    'Perceptions of corruption': 'Perceptions of corruption',
    # 2018/2019
    'Country or region': 'Country',
    'Score': 'Happiness Score',
}

all_dfs = []
for file in files:
    df = pd.read_csv(file)
    # Standardize columns
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # Add missing columns for year
    if 'Year' not in df.columns:
        year = int(file[:4])
        df['Year'] = year
    # For 2018/2019, add Region if missing (set to NaN)
    if 'Region' not in df.columns:
        df['Region'] = np.nan
    # For 2018/2019, rename columns
    if 'Country or region' in df.columns:
        df['Country'] = df['Country or region']
    all_dfs.append(df)

# Concatenate all
merged = pd.concat(all_dfs, ignore_index=True)

# Unify columns for analysis
columns_needed = [
    'Country', 'Region', 'Year', 'Happiness Score', 'GDP per capita', 'Social support',
    'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
]
merged = merged[columns_needed]

# Save merged file
merged.to_csv('world_happiness_report.csv', index=False)

# 2. Clean and Prepare Data
# Check for nulls and drop rows with missing essential values
clean_df = merged.dropna(subset=['Happiness Score', 'GDP per capita', 'Social support',
                                 'Healthy life expectancy', 'Freedom to make life choices',
                                 'Generosity', 'Perceptions of corruption'])

# 3. Analysis
# ==========

# Create output dir
viz_dir = 'vizualization'
os.makedirs(viz_dir, exist_ok=True)

# Task 1: Feature Importance (Linear Regression)
X = clean_df[['GDP per capita', 'Social support', 'Healthy life expectancy',
              'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
y = clean_df['Happiness Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

# Save feature importance plot
plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
plt.title('Feature Importance (Linear Regression)')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
plt.close()

# Task 2: Happiness by Region (Bar plot)
region_avg = clean_df.groupby('Region')['Happiness Score'].mean().sort_values()
plt.figure(figsize=(10,6))
region_avg.plot(kind='barh', color='skyblue')
plt.title('Average Happiness Score by Region')
plt.xlabel('Happiness Score')
plt.ylabel('Region')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'happiness_by_region.png'))
plt.close()

# Task 3: Correlation Heatmap
subset = clean_df[['Happiness Score', 'GDP per capita', 'Freedom to make life choices', 'Perceptions of corruption']]
correlation = subset.corr()
plt.figure(figsize=(7,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation with Happiness Score')
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'correlation_heatmap.png'))
plt.close()

# Optional: Pairplot (save if time/space allows)
sns.pairplot(subset)
plt.savefig(os.path.join(viz_dir, 'pairplot.png'))
plt.close()

# Summary of Insights
summary = f"""
World Happiness Report Analysis (2015-2019)
===========================================

- R² Score for Linear Regression: {r2:.3f}
- Top features predicting happiness: {', '.join(coefficients['Feature'].values[:3])}
- Highest average happiness by region: {region_avg.idxmax()} ({region_avg.max():.2f})
- Lowest average happiness by region: {region_avg.idxmin()} ({region_avg.min():.2f})
- Correlation with happiness: \n{correlation['Happiness Score'].to_string()}

See visualizations in the 'vizualization' folder:
- feature_importance.png
- happiness_by_region.png
- correlation_heatmap.png
- pairplot.png
"""
with open(os.path.join(viz_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)
