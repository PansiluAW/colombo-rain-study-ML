import json,unittest
import numpy as np
from src.data import ROOT, FEATURES, make_frame, splits
from src.predict import INPUTS, load_model, make_input, assess

class PredictionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_model()
        cls.summary=json.loads((ROOT/'results/metrics.json').read_text())

    def test_form_reconstructs_all_test_rows(self):
        test=splits(make_frame())[2]
        frames=[make_input(row.target_date.date(),{key:row[key] for key in INPUTS}) for _,row in test.iterrows()]
        import pandas as pd
        combined=pd.concat(frames,ignore_index=True)
        np.testing.assert_allclose(combined.to_numpy(),test[FEATURES].to_numpy(),atol=1e-12)
        expected=pd.read_csv(ROOT/'results/test_predictions.csv')['Regularised QDA'].to_numpy()
        np.testing.assert_allclose(self.model.predict_proba(combined)[:,1],expected,atol=1e-12)

    def test_explanations_for_examples(self):
        for ex in self.summary['examples']:
            x=make_input(ex['date'],dict(zip(INPUTS,ex['features'])))
            result=assess(self.model,x,.75)
            self.assertAlmostEqual(result['probability'],ex['probability'],places=12)
            self.assertAlmostEqual(result['base']+sum(result['shap']),result['raw'],places=10)

    def test_impossible_weather_rejected(self):
        ex=self.summary['examples'][0]
        values=dict(zip(INPUTS,ex['features']))
        values['temp_min_today']=40
        with self.assertRaises(ValueError):make_input(ex['date'],values)
        values['temp_min_today']=20;values['rain_mean_3d']=0
        with self.assertRaises(ValueError):make_input(ex['date'],values)

if __name__=='__main__':unittest.main()
