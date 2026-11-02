"""Figures from the saved validation, test and SHAP results; no model selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from src.data import ROOT

PLOTS = ROOT / 'results/plots'
SHORT_LABELS = ['Preceding-day rain', 'Humidity', 'Maximum temperature',
                'Minimum temperature', 'Maximum wind', 'Mean pressure',
                '3-day mean rain', '7-day mean rain', 'Pressure change',
                'Season sine', 'Season cosine']


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def run():
    PLOTS.mkdir(exist_ok=True)
    summary = json.loads((ROOT / 'results/metrics.json').read_text())
    predictions = pd.read_csv(ROOT / 'results/test_predictions.csv')
    y = predictions.wet_tomorrow
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    styles = [('#222222', '-'), ('#555555', '--'), ('#888888', '-.'),
              ('#444444', ':'), ('#aaaaaa', '-')]
    for name, xlabel, ylabel, curve, score in [
        ('roc', 'False positive rate', 'True positive rate', roc_curve, 'auc'),
        ('precision_recall', 'Recall (wet days)', 'Precision (wet days)', precision_recall_curve, 'ap'),
    ]:
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        for (model_name, metrics), (color, style) in zip(summary['test'].items(), styles):
            a, b, _ = curve(y, predictions[model_name])
            if name == 'precision_recall':
                a, b = b, a
            ax.plot(a, b, color=color, linestyle=style,
                    label=f'{model_name}: {metrics[score]:.3f}')
        if name == 'roc':
            ax.plot([0, 1], [0, 1], color='.7', linestyle='--', linewidth=.8)
        else:
            ax.axhline(y.mean(), color='.7', linestyle='--', linewidth=.8,
                       label=f'Wet prevalence: {y.mean():.3f}')
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.03))
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(11); ax.yaxis.label.set_size(11)
        ax.legend(fontsize=9.5, loc='lower right' if name == 'roc' else 'lower left')
        save(fig, name)

    candidates = summary['candidates']
    aucs = [item['validation_auc'] for item in candidates]
    selected = summary['selected_params']['reg_param']
    chosen = next(i for i, item in enumerate(candidates) if item['reg_param'] == selected)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(range(len(candidates)), aucs, color='.25', marker='o', markersize=4)
    ax.scatter([chosen], [aucs[chosen]], color='black', s=65, zorder=3)
    ax.annotate(f'Selected: {selected:.2f}; AUC {aucs[chosen]:.4f}',
                (chosen, aucs[chosen]), xytext=(-155, -24), textcoords='offset points', fontsize=9)
    ax.set_xticks(range(len(candidates)), [f'{item["reg_param"]:g}' for item in candidates])
    ax.set(xlabel='Tested regularisation values (alpha)', ylabel='Validation AUC')
    save(fig, 'validation')

    m = summary['test']['Regularised QDA']
    matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(matrix, cmap='Greys', vmin=0, vmax=matrix.max() * 1.2)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha='center', va='center', fontsize=19,
                color='white' if value > matrix.max() * .6 else 'black')
    ax.set_xticks([0, 1], ['Dry', 'Wet']); ax.set_yticks([0, 1], ['Dry', 'Wet'])
    ax.set(xlabel='Predicted class', ylabel='Actual class', title='QDA / cutoff 0.78')
    save(fig, 'confusion')

    importance = pd.read_csv(ROOT / 'results/shap_importance.csv')
    label_map = dict(zip(summary['features'], SHORT_LABELS))
    g = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.barh([label_map[key] for key in g.feature], g.mean_abs_shap, color='.2', height=.62)
    ax.set_xlabel('Mean absolute interventional SHAP (log odds)')
    save(fig, 'shap_global')

    examples = [item for item in summary['examples'] if item['name'] in ['Correct wet day', 'False wet alert']]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, item in zip(axes, examples):
        values = np.array(item['shap'])
        order = np.argsort(np.abs(values))[-5:]
        ax.barh(np.array(SHORT_LABELS)[order], values[order],
                color=['.25' if value >= 0 else '.7' for value in values[order]], height=.6)
        ax.axvline(0, color='.4', linewidth=.7)
        ax.set_title(f'{item["date"]} / {item["name"]}\np(wet) {item["probability"]:.1%}; rain {item["actual_rain"]:.1f} mm', fontsize=11)
        ax.set_xlabel('SHAP (log odds)', fontsize=11)
        ax.tick_params(labelsize=10.5)
    save(fig, 'shap_local')


if __name__ == '__main__':
    run()
