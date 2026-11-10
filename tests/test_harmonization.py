import tempfile
import unittest
from pathlib import Path
import pandas as pd
from merge_happiness_data import FEATURES, load_and_harmonize, grouped_association


class HarmonizationTests(unittest.TestCase):
    def test_region_mapping_and_country_held_out_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for year in range(2015, 2020):
                frame = pd.DataFrame({
                    'Country': [f'C{i}' for i in range(12)],
                    'Region': ['R'] * 12,
                    'Happiness Score': [float(i + year / 100) for i in range(12)],
                    'GDP per capita': range(12),
                    'Social support': range(12),
                    'Healthy life expectancy': range(12),
                    'Freedom to make life choices': range(12),
                    'Generosity': range(12),
                    'Perceptions of corruption': range(12),
                })
                if year >= 2018:
                    frame = frame.drop(columns='Region')
                path = Path(tmp) / f'{year}.csv'
                frame.to_csv(path, index=False)
                files.append(path)
            merged, hashes = load_and_harmonize(files)
            self.assertTrue(merged['Region'].notna().all())
            self.assertEqual(len(hashes), 5)
            _, _, metrics = grouped_association(merged)
            self.assertGreater(metrics['test_countries'], 0)

    def test_2015_inputs_are_components_of_reported_score(self):
        source = pd.read_csv('2015.csv')
        merged, _ = load_and_harmonize([Path('2015.csv')])
        reconstructed = merged[FEATURES].sum(axis=1) + source['Dystopia Residual']
        self.assertLess((merged['Happiness Score'] - reconstructed).abs().max(), 0.001)


if __name__ == '__main__':
    unittest.main()
