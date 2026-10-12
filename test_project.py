import json
import sqlite3
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.forecasting import MODELS,forecast
from src.pipeline import ROOT,metrics

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weekly=pd.read_csv(ROOT/'outputs/weekly_sales.csv',parse_dates=['week'])
        cls.pred=pd.read_csv(ROOT/'outputs/all_rolling_predictions.csv.gz',dtype={'sku':str})
        cls.summary=json.loads((ROOT/'outputs/summary.json').read_text())

    def test_week_coverage_and_training_cohort(self):
        w=self.weekly
        self.assertEqual(len(w),104)
        self.assertEqual(len(w.columns)-1,20)
        self.assertTrue(w.week.diff().dropna().eq(pd.Timedelta(weeks=1)).all())
        self.assertTrue(w.iloc[:,1:].ge(0).all().all())
        cohort=pd.read_csv(ROOT/'outputs/sku_cohort.csv',dtype={'StockCode':str})
        for row in cohort.itertuples():
            self.assertEqual(row.training_units,w[row.StockCode].iloc[:64].sum())
            self.assertGreaterEqual((w[row.StockCode].iloc[:64]>0).sum(),32)
        self.assertEqual(w['84270'].iloc[88:].sum(),0)

    def test_chronological_blocks(self):
        p=self.pred
        v=p.loc[p.phase.eq('selection')]
        t=p.loc[p.phase.eq('test')]
        self.assertEqual(set(v.origin_index),{64,68,72})
        self.assertEqual(set(t.origin_index),{88,92,96,100})
        c=pd.read_csv(ROOT/'outputs/calibration_errors.csv')
        self.assertEqual(set(c.origin_index),{76,80,84})
        self.assertLess((v.origin_index+v.horizon-1).max(),76)
        self.assertLess((c.origin_index+c.protection_weeks-1).max(),88)
        self.assertFalse(p.duplicated(['phase','sku','model','target_week']).any())

    def test_prior_history_reproduces_forecasts(self):
        sku='85123A';y=self.weekly[sku].to_numpy()
        for model in MODELS:
            saved=self.pred.loc[self.pred.sku.eq(sku)&self.pred.origin_index.eq(88)&self.pred.model.eq(model)].sort_values('horizon')
            np.testing.assert_allclose(forecast(y[:88],4,model),saved.predicted,rtol=1e-8)
            changed=y.copy();changed[88:]=1000000
            np.testing.assert_allclose(forecast(changed[:88],4,model),saved.predicted,rtol=1e-8)

    def test_model_selection_and_metrics(self):
        v=pd.read_csv(ROOT/'outputs/selection_scores.csv',dtype={'sku':str})
        expected=v.sort_values(['sku','mae','model_order']).groupby('sku').first().reset_index()
        saved=pd.read_csv(ROOT/'outputs/selected_models.csv',dtype={'sku':str})
        pd.testing.assert_series_equal(expected.model,saved.model,check_names=False)
        global_scores=pd.read_csv(ROOT/'outputs/global_selection_scores.csv')
        self.assertEqual(global_scores.sort_values(['wape','model']).iloc[0].model,self.summary['validation_champion'])
        scores=pd.read_csv(ROOT/'outputs/test_metrics.csv').set_index('model')
        for model,g in self.pred.loc[self.pred.phase.eq('test')].groupby('model'):
            calc=metrics(g,self.summary['series_scales_lag1'])
            for metric,value in calc.items():self.assertAlmostEqual(value,scores.loc[model,metric])

    def test_calibrated_buffers_use_prior_windows(self):
        errors=pd.read_csv(ROOT/'outputs/calibration_errors.csv')
        b=pd.read_csv(ROOT/'outputs/calibrated_buffers.csv')
        for row in b.itertuples():
            g=errors.loc[errors.policy.eq(row.policy)&errors.protection_weeks.eq(row.protection_weeks)]
            self.assertEqual(len(g),60)
            self.assertAlmostEqual(row.scaled_safety_buffer,np.quantile(g.scaled_positive_error,row.empirical_quantile,method='higher'))
            self.assertAlmostEqual(row.scaled_horizon_band,np.quantile(g.scaled_horizon_absolute_error,row.empirical_quantile,method='higher'))

    def test_stock_flow_and_lead_time_conservation(self):
        traces=pd.read_csv(ROOT/'outputs/inventory_trajectories.csv.gz',dtype={'sku':str})
        for (policy,sku,lead,q),g in traces.groupby(['policy','sku','lead_weeks','buffer_quantile']):
            g=g.sort_values('week_index').reset_index(drop=True)
            self.assertEqual(len(g),16)
            np.testing.assert_allclose(g.arrivals,g.ordered.shift(lead).fillna(0))
            np.testing.assert_allclose(g.ending_on_hand,g.opening_after_arrival-g.served_proxy_units)
            np.testing.assert_allclose(g.served_proxy_units+g.unmet_proxy_units,g.observed_sales_driver)
            np.testing.assert_allclose(g.ending_pipeline,g.ordered.cumsum()-g.arrivals.cumsum())
            np.testing.assert_allclose(g.opening_after_arrival.iloc[1:],g.ending_on_hand.shift(1).iloc[1:]+g.arrivals.iloc[1:])
            self.assertEqual(g.opening_after_arrival.iloc[0],g.initial_on_hand.iloc[0])
            self.assertTrue(g[['ending_on_hand','ending_pipeline','ordered','unmet_proxy_units']].ge(0).all().all())

    def test_sql_reconciles_scenarios(self):
        d=pd.read_csv(ROOT/'outputs/inventory_trajectories.csv.gz',dtype={'sku':str})
        con=sqlite3.connect(':memory:');d.to_sql('inventory_trace',con,index=False)
        for block in (ROOT/'analysis.sql').read_text().split('-- report: ')[1:]:
            name,query=block.split('\n',1)
            computed=pd.read_sql_query(query,con)
            saved=pd.read_csv(ROOT/f'outputs/{name.strip()}.csv',dtype={'sku':str})
            pd.testing.assert_frame_equal(computed,saved,check_dtype=False)
        con.close()

    def test_dashboard_and_zero_sales_scenario(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=40).run()
        self.assertFalse(app.exception)
        app.sidebar.selectbox[0].select('84270')
        app.sidebar.selectbox[1].select('Per-SKU selection')
        app.run();self.assertFalse(app.exception)
        app.radio[0].set_value('Selected SKU')
        app.run();self.assertFalse(app.exception)
        app.selectbox[0].select(3)
        app.run();self.assertFalse(app.exception)
        self.assertTrue(any(m.label=='Replay fill' and m.value=='—' for m in app.metric))

if __name__=='__main__':unittest.main()
