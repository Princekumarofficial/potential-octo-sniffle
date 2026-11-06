"""Run with python -m unittest full_predictor.test_feature_engineering."""
import unittest
import numpy as np
import pandas as pd
from BTP.full_predictor.helpers.feature_engineering import add_features, validate_features, daily_garch


class FeatureTests(unittest.TestCase):
    def fixture(self):
        idx = pd.date_range("2025-01-02 09:15",periods=200,freq="min")
        values = np.arange(200,dtype=float)+100
        f = pd.DataFrame(dict(open=values, high=values+1, low=values-1,
                              close=values, volume=100.),index=idx)
        f["is_missing_bar"] = False
        f["is_valid_bar"] = True
        return f

    def test_missing_bar_and_warmup(self):
        f = self.fixture()
        f.loc[f.index[100],["open","high","low","close","volume"]] = np.nan
        f.loc[f.index[100],["is_missing_bar","is_valid_bar"]] = [True,False]
        a,cols,_ = add_features(f,False)
        validate_features(a,cols)
        self.assertTrue(a.loc[f.index[100],cols].isna().all())
        self.assertTrue(pd.isna(a.sma_20.iloc[119]))
        self.assertAlmostEqual(a.sma_20.iloc[120],f.close.iloc[101:121].mean())
        self.assertTrue(pd.isna(a.returns.iloc[101]))

    def test_known_values_and_prefix(self):
        f = self.fixture()
        a,cols,_ = add_features(f,False)
        self.assertAlmostEqual(a.trend_slope_20.iloc[50],1.)
        self.assertEqual(a.rsi_14.iloc[50],100.)
        self.assertAlmostEqual(a.sma_20.iloc[50],np.mean(np.arange(131,151)))
        b,_,_ = add_features(f.iloc[:150],False)
        pd.testing.assert_frame_equal(a.iloc[:150],b)
        validate_features(a,cols)

    def test_zero_denominators(self):
        f = self.fixture()
        f[["open","high","low","close"]] = 100.
        f["volume"] = 0.
        a,cols,_ = add_features(f,False)
        validate_features(a,cols)
        self.assertEqual(a.rsi_14.iloc[50],50.)
        self.assertTrue(a.bb_position_20.isna().all())
        self.assertTrue(a.relative_volume_20.isna().all())

    def test_session_reset(self):
        f = self.fixture()
        f.index = f.index[:100].append(f.index[100:]+pd.Timedelta(days=1))
        a,_,_ = add_features(f,False)
        self.assertTrue(pd.isna(a.returns.iloc[100]))
        self.assertTrue(pd.isna(a.sma_20.iloc[100]))

    def test_garch_uses_prior_days_only(self):
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2020-01-01",periods=180)+pd.Timedelta(hours=15,minutes=29)
        prices = 100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
        f = pd.DataFrame(dict(close=prices,is_valid_bar=True),index=dates)
        a,records = daily_garch(f)
        b,_ = daily_garch(f.iloc[:140])
        pd.testing.assert_series_equal(a.iloc[:140],b)
        self.assertGreater(a.notna().sum(),0)
        self.assertTrue(all(pd.Timestamp(r["fit_end"])<pd.Timestamp(r["forecast_start"]) for r in records))
        changed = f.copy()
        changed.loc[dates[140]:,"close"] *= 2
        c,_ = daily_garch(changed)
        pd.testing.assert_series_equal(a.iloc[:141],c.iloc[:141])


if __name__ == "__main__":
    unittest.main()
