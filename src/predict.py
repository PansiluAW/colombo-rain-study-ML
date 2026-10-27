"""Feature construction shared by the Streamlit form and prediction checks."""
from datetime import date
import math
import numpy as np
import pandas as pd
from src.model import RainQDA
from src.data import ROOT, FEATURES

INPUTS = FEATURES[:9]
LIMITS = [(0,500),(0,100),(10,50),(5,45),(0,200),(950,1050),(0,500),(0,500),(-50,50)]

def make_input(target_date, values):
    if isinstance(target_date,str):
        target_date=date.fromisoformat(target_date)
    if not isinstance(target_date,date):
        raise ValueError('Choose a valid date to predict.')
    numbers=[float(values[key]) for key in INPUTS]
    for key,value,(low,high) in zip(INPUTS,numbers,LIMITS):
        if not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{key}: enter a value from {low} to {high}.')
    rain,humidity,maximum,minimum,wind,pressure,rain3,rain7,change=numbers
    if minimum>maximum:
        raise ValueError('Minimum temperature cannot be above maximum temperature.')
    if 3*rain3+0.02<rain:
        raise ValueError('The three-day rain total must include the preceding day\'s rain.')
    if 7*rain7+0.02<3*rain3:
        raise ValueError('The seven-day rain total must include the three-day total.')
    angle=2*math.pi*(target_date.timetuple().tm_yday-1)/365.25
    return pd.DataFrame([numbers+[math.sin(angle),math.cos(angle)]],columns=FEATURES)

def load_model():
    return RainQDA.load(ROOT/'models/qda.json')

def assess(model, frame, threshold):
    probability=float(model.predict_proba(frame)[0,1])
    raw=float(model.decision_function(frame)[0])
    contributions,base=model.explain(frame)
    shap=contributions[0]
    if not np.isclose(base+shap.sum(),raw,atol=1e-8):
        raise RuntimeError('The explanation did not reconstruct the model score.')
    return dict(probability=probability,label='WET' if probability>=threshold else 'DRY',
                raw=raw,base=base,shap=shap)
