"""Evaluate the frozen model without reselecting any setting on test data."""
import json
from src.predict import load_model
from src.data import ROOT, FEATURES, make_frame, splits
from src.train import metrics

if __name__ == '__main__':
    model=load_model()
    recorded=json.loads((ROOT/'results/metrics.json').read_text())
    test=splits(make_frame())[2]
    result=metrics(test.wet_tomorrow,model.predict_proba(test[FEATURES])[:,1],recorded['test']['Regularised QDA']['threshold'])
    for name,value in result.items():
        assert abs(value-recorded['test']['Regularised QDA'][name])<1e-10, name
    print(json.dumps(result,indent=2))
