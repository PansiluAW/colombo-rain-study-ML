"""Colombo rain notebook. Run locally with flask --app app run."""
from datetime import date
import json

from flask import Flask, abort, render_template, request, send_from_directory
import numpy as np
import pandas as pd

from src.data import ROOT, FEATURES, LABELS
from src.predict import INPUTS, LIMITS, assess, load_model, make_input

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
model = load_model()
summary = json.loads((ROOT / 'results/metrics.json').read_text())
data = pd.read_csv(ROOT / 'data/model_dataset.csv')
examples = summary['examples']
threshold = summary['test']['Regularised QDA']['threshold']
train = data[data.target_date < '2022-01-01']

FIELDS = [
    ('Rain, preceding day', 'mm'), ('Mean humidity', '%'),
    ('Maximum temperature', '°C'), ('Minimum temperature', '°C'),
    ('Maximum wind', 'km/h'), ('Mean pressure', 'hPa'),
    ('3-day mean rain', 'mm/day'), ('7-day mean rain', 'mm/day'),
    ('Pressure change', 'hPa'),
]
CHARTS = {'validation', 'roc', 'precision_recall', 'confusion', 'shap_global', 'shap_local'}


@app.route('/', methods=['GET', 'POST'])
def notebook():
    error = None
    prediction = None
    matched = None
    contributions = []
    outside = []
    if request.method == 'POST':
        values = {key: request.form.get(key, '') for key in INPUTS}
        target = request.form.get('target_date', '')
    else:
        try:
            index = int(request.args.get('example', '0'))
            if not 0 <= index < len(examples):
                raise ValueError
        except ValueError:
            abort(404)
        example = examples[index]
        values = dict(zip(INPUTS, example['features']))
        target = example['date']
    try:
        frame = make_input(target, values)
        prediction = assess(model, frame, threshold)
        target_label = date.fromisoformat(target).strftime('%d %b %Y')
        matched = next((item for item in examples if item['date'] == target
                       and np.allclose(frame.iloc[0].to_numpy(), item['features'],
                                       atol=1e-6, rtol=0)), None)
        outside = [LABELS[i] for i, key in enumerate(INPUTS)
                   if not train[key].min() <= frame.iloc[0][key] <= train[key].max()]
        largest = max(float(np.abs(prediction['shap']).max()), 1e-12)
        for i in np.argsort(np.abs(prediction['shap']))[::-1]:
            value = float(prediction['shap'][i])
            contributions.append(dict(label=LABELS[i], value=value,
                                      width=round(100 * abs(value) / largest, 3)))
    except (ValueError, TypeError, OverflowError) as exc:
        error = str(exc)
        target_label = target
    # Keep recorded rolling means readable without changing the saved dataset.
    shown = {key: f'{value:.8g}' if isinstance(value, (int, float)) else value
             for key, value in values.items()}
    fields = [dict(key=key, label=label, unit=unit, low=low, high=high,
                   value=shown[key])
              for key, (label, unit), (low, high) in zip(INPUTS, FIELDS, LIMITS)]
    return render_template('notebook.html', active='entry', fields=fields,
                           target=target, target_label=target_label, examples=examples,
                           prediction=prediction, threshold=threshold, matched=matched,
                           contributions=contributions, outside=outside, error=error), 422 if error else 200


@app.get('/results')
def results():
    return render_template('results.html', active='results', summary=summary)


@app.get('/data')
def dataset():
    return render_template('data.html', active='data')


@app.get('/data/daily.csv')
def download_data():
    return send_from_directory(ROOT / 'data', 'colombo_daily.csv', as_attachment=True)


@app.get('/figures/<name>.png')
def figure(name):
    if name not in CHARTS:
        abort(404)
    return send_from_directory(ROOT / 'results/plots', name + '.png')


@app.get('/health')
def health():
    return {'status': 'ok'}
