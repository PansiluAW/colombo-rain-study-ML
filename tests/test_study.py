"""Checks for data alignment, exported scores, SHAP and the submitted frontend."""
import hashlib
import itertools
import json
import math
import unittest
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from app import app, examples, summary
from src.data import ROOT, FEATURES, make_frame, splits
from src.predict import INPUTS, assess, load_model, make_input


class StudyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = make_frame()
        cls.model = load_model()

    def test_source_and_target_alignment(self):
        provenance = json.loads((ROOT / 'data/provenance.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'data/colombo_era5_raw.json').read_bytes()).hexdigest(), provenance['raw_sha256'])
        raw = pd.read_csv(ROOT / 'data/colombo_daily.csv')
        self.assertEqual(len(raw), 4383)
        self.assertEqual(len(self.frame), 4376)
        np.testing.assert_allclose(self.frame.target_rain_mm, raw.precipitation_sum.iloc[7:].to_numpy())
        np.testing.assert_allclose(self.frame.rain_mean_7d, raw.precipitation_sum.rolling(7).mean().iloc[6:-1].to_numpy())
        self.assertTrue(((self.frame.target_date - self.frame.observation_date).dt.days == 1).all())
        self.assertNotIn('target_rain_mm', FEATURES)

    def test_chronological_splits_and_validation_selection(self):
        train, validation, test = splits(self.frame)
        self.assertEqual([len(train), len(validation), len(test)], [2915, 730, 731])
        self.assertLess(train.target_date.max(), validation.target_date.min())
        self.assertLess(validation.target_date.max(), test.target_date.min())
        chosen = max(summary['candidates'], key=lambda item: item['validation_auc'])
        self.assertEqual(chosen['reg_param'], .75)
        self.assertEqual(chosen['reg_param'], summary['selected_params']['reg_param'])

    def test_all_saved_test_predictions(self):
        test = splits(self.frame)[2]
        exported = pd.read_csv(ROOT / 'results/test_predictions.csv')
        np.testing.assert_allclose(self.model.predict_proba(test[FEATURES])[:, 1], exported['Regularised QDA'], rtol=1e-12, atol=1e-12)
        shap, base = self.model.explain(test[FEATURES])
        np.testing.assert_allclose(base + shap.sum(axis=1), self.model.decision_function(test[FEATURES]), atol=1e-10)

    def test_shap_against_all_coalitions(self):
        # An independent Shapley calculation using the empirical training background.
        training = splits(self.frame)[0][FEATURES]
        background = self.model.transform(training)
        x = self.model.transform(self.frame[FEATURES].iloc[[-1]])[0]
        n = len(FEATURES)
        values = {}
        for bits in itertools.product([False, True], repeat=n):
            mask = np.array(bits)
            hybrid = background.copy()
            hybrid[:, mask] = x[mask]
            scores = self.model.intercept + hybrid @ self.model.linear + np.einsum('ni,ij,nj->n', hybrid, self.model.quadratic, hybrid)
            values[sum(int(bit) << i for i, bit in enumerate(bits))] = float(scores.mean())
        expected = np.zeros(n)
        for i in range(n):
            for mask in range(1 << n):
                if mask & (1 << i):
                    continue
                size = mask.bit_count()
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                expected[i] += weight * (values[mask | (1 << i)] - values[mask])
        actual, base = self.model.explain(self.frame[FEATURES].iloc[[-1]])
        np.testing.assert_allclose(actual[0], expected, atol=1e-10)
        self.assertAlmostEqual(base, values[0], places=10)


class FrontendChecks(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.values = dict(zip(INPUTS, examples[1]['features']))
        self.values['target_date'] = examples[1]['date']

    def test_pages_figures_and_download(self):
        for path in ['/', '/results', '/data', '/health', '/data/daily.csv',
                     '/static/notebook.css', '/static/interaction.js'] + [f'/figures/{name}.png' for name in
                     ['validation', 'roc', 'precision_recall', 'confusion', 'shap_global', 'shap_local']]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                response.close()
        self.assertEqual(self.client.get('/health').json, {'status': 'ok'})

    def test_all_four_recorded_examples(self):
        for i, example in enumerate(examples):
            response = self.client.get(f'/?example={i}')
            html = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(f'{100 * example["probability"]:.1f}%', html)
            self.assertIn('Recorded next-day rain', html)

    def test_changed_input_recomputes_prediction_and_explanation(self):
        original = self.client.post('/', data=self.values).get_data(as_text=True)
        self.assertIn('29.6%', original)
        self.values['humidity_today'] = 86
        response = self.client.post('/', data=self.values)
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('62.0%', html)
        self.assertIn('>DRY<', html)
        self.assertNotIn('Recorded next-day rain', html)
        frame = make_input(self.values['target_date'], self.values)
        expected = assess(load_model(), frame, .78)
        for value in expected['shap']:
            self.assertIn(f'{value:+.4f}', html)

    def test_comparison_uses_the_recorded_reference_and_can_be_restored(self):
        values = self.values | {'humidity_today': 86, 'reference_example': '1',
                                'reference_probability': '0'}
        html = self.client.post('/', data=values).get_data(as_text=True)
        self.assertIn('29.6%', html)
        self.assertIn('62.0%', html)
        self.assertIn('+32.4 percentage points', html)
        self.assertIn(examples[1]['name'], html)
        self.assertNotIn('Recorded next-day rain', html)
        self.assertIn('href="/?example=1">Restore example', html)
        restored = self.client.get('/?example=1').get_data(as_text=True)
        self.assertIn('29.6%', restored)
        self.assertIn('Recorded next-day rain', restored)
        self.assertNotIn('Change from original entry', restored)

    def test_invalid_comparison_references_are_rejected(self):
        for reference in ['-1', '4', 'invalid', '<script>']:
            with self.subTest(reference=reference):
                response = self.client.post('/', data=self.values | {'reference_example': reference})
                self.assertEqual(response.status_code, 422)
                self.assertIn('Choose a recorded example', response.get_data(as_text=True))
                self.assertNotIn('class="probability"', response.get_data(as_text=True))

    def test_all_contribution_bars_and_full_size_charts_are_available(self):
        class Page(HTMLParser):
            def __init__(self):
                super().__init__()
                self.items = 0
                self.charts = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                self.items += tag == 'li'
                if tag == 'a' and attrs.get('href', '').startswith('/figures/'):
                    self.charts.append(attrs['href'])
        entry = Page()
        entry.feed(self.client.get('/').get_data(as_text=True))
        self.assertEqual(entry.items, 11)
        results = Page()
        results.feed(self.client.get('/results').get_data(as_text=True))
        self.assertEqual(set(results.charts), {f'/figures/{name}.png' for name in
                         ['validation', 'roc', 'precision_recall', 'confusion', 'shap_global', 'shap_local']})

    def test_invalid_entries_do_not_predict(self):
        cases = [dict(temp_min_today=40), dict(rain_today=-1),
                 dict(humidity_today='nan'), dict(wind_today='inf'),
                 dict(rain_today=50, rain_mean_3d=1),
                 dict(rain_mean_3d=20, rain_mean_7d=1),
                 dict(target_date='2025-02-30'), dict(pressure_today='')]
        for changes in cases:
            with self.subTest(changes=changes):
                response = self.client.post('/', data=self.values | changes)
                self.assertEqual(response.status_code, 422)
                self.assertIn('Check the readings', response.get_data(as_text=True))
                self.assertNotIn('class="result-word"', response.get_data(as_text=True))

    def test_outside_training_warning(self):
        response = self.client.post('/', data=self.values | {'pressure_today': 950})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Outside the training range', response.get_data(as_text=True))

    def test_requests_do_not_share_form_state(self):
        self.client.post('/', data=self.values | {'humidity_today': 86})
        other = app.test_client().get('/?example=1').get_data(as_text=True)
        self.assertIn('29.6%', other)

    def test_missing_fields_and_escaping(self):
        response = self.client.post('/', data={'target_date': '<script>alert(1)</script>'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('<script>', response.get_data(as_text=True))

    def test_unknown_examples_and_files(self):
        for path in ['/?example=-1', '/?example=99', '/?example=abc', '/figures/unknown.png']:
            self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == '__main__':
    unittest.main()
