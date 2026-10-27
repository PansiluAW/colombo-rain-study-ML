"""Check the score against sklearn and explanations against the coalition game."""
import math
import unittest
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from src.data import FEATURES, make_frame, splits
from src.model import RainQDA, transform_rain
from src.predict import load_model


class QDAChecks(unittest.TestCase):
    def test_portable_model_matches_sklearn(self):
        model = load_model()
        train, validation, test = splits(make_frame())
        scaler = StandardScaler().fit(transform_rain(train[FEATURES]))
        fitted = QuadraticDiscriminantAnalysis(reg_param=model.parameters['reg_param'])
        fitted.fit(scaler.transform(transform_rain(train[FEATURES])), train.wet_tomorrow)
        for rows in (validation, test):
            transformed = scaler.transform(transform_rain(rows[FEATURES]))
            np.testing.assert_allclose(model.decision_function(rows[FEATURES]),
                fitted.decision_function(transformed), atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(model.predict_proba(rows[FEATURES]),
                fitted.predict_proba(transformed), atol=1e-12)

    def test_shap_matches_exhaustive_interventional_game(self):
        model = load_model()
        train, _, test = splits(make_frame())
        background = train[FEATURES].iloc[np.linspace(0, len(train)-1, 32, dtype=int)].to_numpy()
        parameters = dict(model.parameters)
        z = model.transform(background)
        parameters['background_mean'] = z.mean(axis=0).tolist()
        parameters['background_second'] = (z.T @ z / len(z)).tolist()
        model = RainQDA(parameters)
        entry = test[FEATURES].iloc[17].to_numpy()
        d = len(FEATURES)
        game = np.empty(2 ** d)
        for mask in range(2 ** d):
            hybrid = background.copy()
            present = [i for i in range(d) if mask & (1 << i)]
            hybrid[:, present] = entry[present]
            game[mask] = model.decision_function(hybrid).mean()
        exhaustive = np.zeros(d)
        for i in range(d):
            for mask in range(2 ** d):
                if mask & (1 << i):
                    continue
                size = mask.bit_count()
                weight = 1 / (d * math.comb(d-1, size))
                exhaustive[i] += weight * (game[mask | (1 << i)] - game[mask])
        actual, base = model.explain(entry.reshape(1, -1))
        np.testing.assert_allclose(actual[0], exhaustive, atol=1e-10)
        self.assertAlmostEqual(base, game[0], places=10)


if __name__ == '__main__':
    unittest.main()
