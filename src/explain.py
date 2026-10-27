"""Recompute exact QDA SHAP and verify raw-score reconstruction."""
import numpy as np
from src.predict import load_model
from src.data import FEATURES, make_frame, splits

if __name__ == '__main__':
    model=load_model()
    test=splits(make_frame())[2]; idx=np.linspace(0,len(test)-1,240,dtype=int)
    x=test[FEATURES].iloc[idx]
    shap,base=model.explain(x)
    assert np.allclose(base+shap.sum(axis=1),model.decision_function(x),atol=1e-8)
    for name,value in sorted(zip(FEATURES,np.abs(shap).mean(axis=0)),key=lambda item:-item[1]):
        print(f'{name:22} {value:.6f}')
