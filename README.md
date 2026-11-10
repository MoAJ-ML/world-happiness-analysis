# World Happiness Report CSV Analysis (2015–2019)

This repository harmonizes five annual CSV files and explores associations with reported happiness scores. Associations are not causal effects. The CSVs' original publisher, extraction method, revisions, and redistribution rights are **not documented in this repository**; the [official World Happiness Report](https://worldhappiness.report/) is context, not verified provenance for these exact files. Hashes of the committed inputs are recorded by the script for reproducibility. Do not treat the repo's MIT software license as a license for third-party data.

## Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
MPLBACKEND=Agg python merge_happiness_data.py
python -m unittest discover -s tests
```

The script maps changing column names, fills missing 2018/2019 regions from earlier country mappings where available, and writes `world_happiness_report.csv`. It standardizes predictors using training countries only, checks a held-out set of countries, and reports R²/MAE in `vizualization/validation_results.json`. Plots show standardized linear coefficients and descriptive correlations. A country-held-out split still does not test later years, and unobserved social factors can confound the associations. The earlier `feature_importance.png`, `pairplot.png`, and summary text are historical outputs under the previous method.

The 2026-11-10 rerun on the five committed CSVs produced country-held-out R² 0.779 and MAE 0.409 score units across 34 test countries. The JSON includes the five input hashes and exact split counts; these results are descriptive associations only.

Suggested repository description: “Harmonized 2015–2019 happiness CSV analysis with country-held-out regression and explicit provenance limits.”
