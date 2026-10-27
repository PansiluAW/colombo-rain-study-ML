import unittest
import hashlib,json
import pandas as pd
from src.data import ROOT,FEATURES,make_frame,splits

class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=make_frame(); cls.raw=pd.read_csv(ROOT/'data/colombo_daily.csv')

    def test_snapshot_hash(self):
        expected=json.loads((ROOT/'data/provenance.json').read_text())['raw_sha256']
        self.assertEqual(hashlib.sha256((ROOT/'data/colombo_era5_raw.json').read_bytes()).hexdigest(),expected)

    def test_target_alignment(self):
        mapping=self.raw.set_index('time').precipitation_sum
        for row in self.d.itertuples():
            rain=mapping[str(row.target_date.date())]
            self.assertEqual(row.target_rain_mm,rain)
            self.assertEqual(row.wet_tomorrow,int(rain>=1))
            self.assertEqual((row.target_date-row.observation_date).days,1)

    def test_windows_use_past_only(self):
        for i in [0,100,2914,2915,3644,3645,len(self.d)-1]:
            row=self.d.iloc[i];raw_index=self.raw.index[self.raw.time==str(row.observation_date.date())][0]
            self.assertAlmostEqual(row.rain_mean_3d,self.raw.precipitation_sum.iloc[raw_index-2:raw_index+1].mean())
            self.assertAlmostEqual(row.rain_mean_7d,self.raw.precipitation_sum.iloc[raw_index-6:raw_index+1].mean())

    def test_time_split(self):
        train,val,test=splits(self.d)
        self.assertEqual([len(train),len(val),len(test)],[2915,730,731])
        self.assertLess(train.target_date.max(),val.target_date.min())
        self.assertLess(val.target_date.max(),test.target_date.min())

    def test_numeric_complete(self):
        self.assertEqual(len(FEATURES),11)
        self.assertTrue(self.d[FEATURES].notna().all().all())
        self.assertTrue((self.d.temp_min_today<=self.d.temp_max_today).all())

if __name__=='__main__':unittest.main()
