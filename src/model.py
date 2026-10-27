"""Portable QDA scores and exact interventional SHAP for a quadratic score.

The background is the empirical training distribution. Coordinates not in a
coalition are taken jointly from a background row. Fixed and missing features
are therefore independent across that boundary, not conditionally sampled.
"""
import json
import numpy as np
from src.data import FEATURES

RAIN_INDICES = [0, 6, 7]


def transform_rain(values):
    values = np.asarray(values, dtype=float).copy()
    values[:, RAIN_INDICES] = np.log1p(values[:, RAIN_INDICES])
    return values


class RainQDA:
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean = np.asarray(parameters['mean'])
        self.scale = np.asarray(parameters['scale'])
        self.quadratic = np.asarray(parameters['quadratic'])
        self.linear = np.asarray(parameters['linear'])
        self.intercept = float(parameters['intercept'])
        self.background_mean = np.asarray(parameters['background_mean'])
        self.background_second = np.asarray(parameters['background_second'])

    @classmethod
    def from_fitted(cls, scaler, qda, training):
        precision = [(r / s) @ r.T for r, s in zip(qda.rotations_, qda.scalings_)]
        p0, p1 = precision
        m0, m1 = qda.means_
        a = .5 * (p0 - p1)
        b = p1 @ m1 - p0 @ m0
        c = (np.log(qda.priors_[1] / qda.priors_[0])
             - .5 * (np.log(qda.scalings_[1]).sum() - np.log(qda.scalings_[0]).sum())
             - .5 * (m1 @ p1 @ m1 - m0 @ p0 @ m0))
        z = scaler.transform(transform_rain(training))
        return cls(dict(features=FEATURES, log1p_indices=RAIN_INDICES,
            reg_param=float(qda.reg_param), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
            quadratic=a.tolist(), linear=b.tolist(), intercept=float(c),
            background_n=len(z), background_mean=z.mean(axis=0).tolist(),
            background_second=(z.T @ z / len(z)).tolist()))

    def transform(self, values):
        return (transform_rain(values) - self.mean) / self.scale

    def decision_function(self, values):
        z = self.transform(values)
        return self.intercept + z @ self.linear + np.einsum('ni,ij,nj->n', z, self.quadratic, z)

    def predict_proba(self, values):
        score = self.decision_function(values)
        # Stable logistic transform, including very large hypothetical inputs.
        exp = np.exp(-np.abs(score))
        wet = np.where(score >= 0, 1 / (1 + exp), exp / (1 + exp))
        return np.column_stack([1 - wet, wet])

    def explain(self, values):
        z = self.transform(values)
        mu, second, a = self.background_mean, self.background_second, self.quadratic
        base = self.intercept + self.linear @ mu + np.sum(a * second)
        phi = (z - mu) * self.linear + (z ** 2 - np.diag(second)) * np.diag(a)
        for i in range(z.shape[1]):
            for j in range(i + 1, z.shape[1]):
                product = z[:, i] * z[:, j] - second[i, j]
                asymmetry = z[:, i] * mu[j] - mu[i] * z[:, j]
                phi[:, i] += a[i, j] * (product + asymmetry)
                phi[:, j] += a[i, j] * (product - asymmetry)
        return phi, float(base)

    def save(self, path):
        path.write_text(json.dumps(self.parameters, indent=2))

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text()))
