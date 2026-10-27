import unittest
import json
from datetime import date
from src.data import ROOT
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'streamlit_app.py'

class AppChecks(unittest.TestCase):
    def test_recorded_dry_example(self):
        app=AppTest.from_file(str(APP),default_timeout=30).run()
        self.assertFalse(app.exception)
        summary=json.loads((ROOT/'results/metrics.json').read_text())
        example=next(x for x in summary['examples'] if x['name']=='Correct dry day')
        label=f'{date.fromisoformat(example["date"]):%d %b %Y} / correct dry day'
        app.selectbox('example').select(label).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('DRY' in block.value for block in app.markdown))
        self.assertTrue(any(f'{example["probability"]:.1%}' in block.value for block in app.markdown))

    def test_invalid_entry_does_not_show_prediction(self):
        app=AppTest.from_file(str(APP),default_timeout=30).run()
        app.number_input('temp_min_today').set_value(40)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Minimum temperature' in error.value for error in app.error))
        self.assertFalse(any('result-word' in block.value and '<div' in block.value and 'WET</div>' in block.value for block in app.markdown))

if __name__=='__main__':unittest.main()
