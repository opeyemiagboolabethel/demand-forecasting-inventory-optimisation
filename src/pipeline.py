"""Selection, calibration and untouched-test forecasting; explicit inventory trace scenarios."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from src.forecasting import MODELS,forecast,simulate
ROOT=Path(__file__).resolve().parents[1]

def metrics(frame,scale):
    actual=frame.actual.to_numpy();error=frame.predicted.to_numpy()-actual
    return {'mae':float(np.abs(error).mean()),'wape':float(np.abs(error).sum()/actual.sum()) if actual.sum() else None,'bias_ratio':float(error.sum()/actual.sum()) if actual.sum() else None,'mase_lag1':float((np.abs(error)/frame.sku.map(scale).to_numpy()).mean())}

def main():
    weekly=pd.read_csv(ROOT/'outputs/weekly_sales.csv',parse_dates=['week'])
    dates=weekly.week;skus=list(weekly.columns[1:]);matrix=weekly.set_index('week')
    scales={sku:float(np.abs(np.diff(matrix[sku].iloc[:64])).mean()) for sku in skus}
    means={sku:float(matrix[sku].iloc[:64].mean()) for sku in skus}
    if any(v<=0 for v in scales.values()) or any(v<=0 for v in means.values()):raise ValueError('Invalid training scale')
    all_predictions=[]
    for phase,origins in [('selection',range(64,76,4)),('test',range(88,104,4))]:
        for sku in skus:
            y=matrix[sku].to_numpy()
            for origin in origins:
                for model in MODELS:
                    pred=forecast(y[:origin],4,model)
                    for h in range(4):
                        all_predictions.append({'phase':phase,'sku':sku,'origin_index':origin,'origin_week':str(dates.iloc[origin].date()),'target_week':str(dates.iloc[origin+h].date()),'horizon':h+1,'model':model,'actual':float(y[origin+h]),'predicted':float(pred[h])})
    predictions=pd.DataFrame(all_predictions)
    selection=predictions.loc[predictions.phase.eq('selection')]
    rows=[]
    for (sku,model),g in selection.groupby(['sku','model']):rows.append({'sku':sku,'model':model,'model_order':MODELS.index(model),**metrics(g,scales)})
    selection_scores=pd.DataFrame(rows)
    winners=selection_scores.sort_values(['sku','mae','model_order']).groupby('sku').first().reset_index()
    chosen=dict(zip(winners.sku,winners.model))
    global_scores=pd.DataFrame([{'model':model,**metrics(g,scales)} for model,g in selection.groupby('model')])
    global_scores.to_csv(ROOT/'outputs/global_selection_scores.csv',index=False)
    champion=str(global_scores.sort_values(['wape','model']).iloc[0].model)
    winners.to_csv(ROOT/'outputs/selected_models.csv',index=False)
    selection_scores.to_csv(ROOT/'outputs/selection_scores.csv',index=False)
    calibration=[]
    # Calibrate each policy forecast before examining the final test; do not promise coverage.
    policies=list(dict.fromkeys(['Selected per SKU',champion,MODELS[1]]))
    for policy in policies:
        for sku in skus:
            y=matrix[sku].to_numpy();model=chosen[sku] if policy==policies[0] else policy
            for origin in range(76,88,4):
                pred=forecast(y[:origin],4,model)
                for protection in range(1,5):
                    actual=y[origin:origin+protection].sum();point=pred[:protection].sum()
                    calibration.append({'policy':policy,'sku':sku,'origin_index':origin,'protection_weeks':protection,'actual_total':float(actual),'point_total':float(point),'scaled_absolute_error':float(abs(actual-point)/means[sku]),'scaled_positive_error':float(max(0,actual-point)/means[sku]),'scaled_horizon_absolute_error':float(abs(y[origin+protection-1]-pred[protection-1])/means[sku])})
    cal=pd.DataFrame(calibration)
    cal.to_csv(ROOT/'outputs/calibration_errors.csv',index=False)
    buffers=[]
    for (policy,protection),g in cal.groupby(['policy','protection_weeks']):
        for level in [.8,.9,.95]:
            buffers.append({'policy':policy,'protection_weeks':int(protection),'empirical_quantile':level,'scaled_safety_buffer':float(np.quantile(g.scaled_positive_error,level,method='higher')),'scaled_absolute_band':float(np.quantile(g.scaled_absolute_error,level,method='higher')),'scaled_horizon_band':float(np.quantile(g.scaled_horizon_absolute_error,level,method='higher')),'calibration_windows':len(g)})
    buffer_table=pd.DataFrame(buffers);buffer_table.to_csv(ROOT/'outputs/calibrated_buffers.csv',index=False)
    test=predictions.loc[predictions.phase.eq('test')].copy()
    test_selected=test.loc[test.apply(lambda r:r.model==chosen[r.sku],axis=1)].copy()
    report=[]
    for model,g in test.groupby('model'):report.append({'model':model,**metrics(g,scales)})
    report.append({'model':'Selected per SKU',**metrics(test_selected,scales)})
    pd.DataFrame(report).to_csv(ROOT/'outputs/test_metrics.csv',index=False)
    band_coefficients=buffer_table.loc[buffer_table.policy.eq(policies[0]) & buffer_table.empirical_quantile.eq(.9)].set_index('protection_weeks').scaled_horizon_band.to_dict()
    test_selected['band_width']=test_selected.apply(lambda r:band_coefficients[r.horizon]*means[r.sku],axis=1)
    test_selected['lower']=(test_selected.predicted-test_selected.band_width).clip(lower=0)
    test_selected['upper']=test_selected.predicted+test_selected.band_width
    test_selected.to_csv(ROOT/'outputs/test_selected_predictions.csv',index=False)
    predictions.to_csv(ROOT/'outputs/all_rolling_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    coverage=[]
    for (sku,origin),g in test_selected.groupby(['sku','origin_index']):
        g=g.sort_values('horizon')
        for protection in range(1,5):
            point=g.predicted.iloc[:protection].sum();actual=g.actual.iloc[:protection].sum()
            b=buffer_table.loc[buffer_table.policy.eq(policies[0]) & buffer_table.protection_weeks.eq(protection) & buffer_table.empirical_quantile.eq(.9)].iloc[0]
            width=b.scaled_absolute_band*means[sku]
            coverage.append({'sku':sku,'origin_index':int(origin),'protection_weeks':protection,'actual_total':float(actual),'point_total':float(point),'lower':float(max(0,point-width)),'upper':float(point+width),'inside_empirical_band':bool(max(0,point-width)<=actual<=point+width)})
    coverage_table=pd.DataFrame(coverage)
    coverage_table.to_csv(ROOT/'outputs/test_band_coverage.csv',index=False)
    cached={}
    for policy in policies:
        for sku in skus:
            y=matrix[sku].to_numpy();model=chosen[sku] if policy==policies[0] else policy
            cached[policy,sku]={t:forecast(y[:t],4,model) for t in range(88,104)}
    trajectories=[]
    for policy in policies:
        for lead in [1,2,3]:
            for level in [.8,.9,.95]:
                coefficient=float(buffer_table.loc[buffer_table.policy.eq(policy) & buffer_table.protection_weeks.eq(lead+1) & buffer_table.empirical_quantile.eq(level),'scaled_safety_buffer'].iloc[0])
                for sku in skus:
                    y=matrix[sku].to_numpy();initial=float(np.ceil(y[80:88].mean()*(lead+1)))
                    rows=simulate(y,cached[policy,sku],coefficient*means[sku],lead,88,104,initial)
                    for row in rows:
                        row.update({'policy':policy,'sku':sku,'lead_weeks':lead,'buffer_quantile':level,'week':str(dates.iloc[row['week_index']].date()),'initial_on_hand':initial})
                    trajectories.extend(rows)
    simulations=pd.DataFrame(trajectories)
    simulations.to_csv(ROOT/'outputs/inventory_trajectories.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    connection=sqlite3.connect(ROOT/'outputs/forecasting.sqlite')
    simulations.to_sql('inventory_trace',connection,if_exists='replace',index=False)
    for block in (ROOT/'analysis.sql').read_text().split('-- report: ')[1:]:
        name,query=block.split('\n',1);pd.read_sql_query(query,connection).to_csv(ROOT/f'outputs/{name.strip()}.csv',index=False)
    connection.close()
    future=[]
    for policy in ['Validation champion','Per-SKU selection']:
        buffer_policy=champion if policy=='Validation champion' else policies[0]
        bands=buffer_table.loc[buffer_table.policy.eq(buffer_policy) & buffer_table.empirical_quantile.eq(.9)].set_index('protection_weeks').scaled_horizon_band.to_dict()
        for sku in skus:
            selected_model=champion if policy=='Validation champion' else chosen[sku]
            pred=forecast(matrix[sku],4,selected_model)
            for h,v in enumerate(pred):future.append({'policy':policy,'sku':sku,'week':str((dates.iloc[-1]+pd.Timedelta(weeks=h+1)).date()),'model':selected_model,'point':float(v),'lower':float(max(0,v-bands[h+1]*means[sku])),'upper':float(v+bands[h+1]*means[sku])})
    pd.DataFrame(future).to_csv(ROOT/'outputs/historical_end_forecast.csv',index=False)
    data_summary=json.loads((ROOT/'outputs/data_summary.json').read_text())
    data_summary.update({'models':MODELS,'validation_champion':champion,'global_selection_scores':global_scores.to_dict('records'),'selected_model_counts':winners.model.value_counts().to_dict(),'test_metrics':report,'test_four_week_empirical_band_coverage':float(coverage_table.loc[coverage_table.protection_weeks.eq(4),'inside_empirical_band'].mean()),'inventory_scenarios':27,'inventory_weeks_per_sku':16,'series_scales_lag1':scales,'series_training_means':means})
    (ROOT/'outputs/summary.json').write_text(json.dumps(data_summary,indent=2,allow_nan=False))
    print(json.dumps(data_summary,indent=2),flush=True)

if __name__=='__main__':main()
