"""Historical forecasting evidence and clearly labelled stock-policy trace scenarios."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Demand & Inventory Intelligence | Opeyemi Agboola',page_icon='📦',layout='wide')
st.markdown('''<style>.stApp{background:#f7f7f2;color:#132d3a}.block-container{max-width:1440px;padding-top:2.5rem;padding-bottom:3rem}.hero{background:linear-gradient(120deg,#0b2838,#146d73);padding:2.8rem;border-radius:22px;color:white;margin-bottom:1.3rem}.hero h1{font-size:clamp(2rem,3vw,3.5rem);line-height:1.12;color:white;margin:1rem 0}.hero p{color:#c9e4e3;font-size:1.05rem;max-width:860px}.label{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase}[data-testid="stMetric"]{background:white;border:1px solid #d6e3e2;padding:1.1rem;border-radius:16px}button[data-baseweb="tab"]{font-size:1rem}</style>''',unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="label">Opeyemi Agboola · Commercial Data Science</div><h1>Demand forecasts.<br>Visible inventory trade-offs.</h1><p>Real retail sales, chronological backtests and an explicit stock-policy laboratory. Inspect what generalizes, what fails, and what a replenishment assumption changes.</p></div>''',unsafe_allow_html=True)

@st.cache_data
def load():
    output=ROOT/'outputs'
    read=lambda name:pd.read_csv(output/name,dtype={'sku':str,'StockCode':str})
    return {'summary':json.loads((output/'summary.json').read_text()),'weekly':read('weekly_sales.csv'),'cohort':read('sku_cohort.csv'),'predictions':read('all_rolling_predictions.csv.gz'),'future':read('historical_end_forecast.csv'),'buffers':read('calibrated_buffers.csv'),'policies':read('inventory_policy_metrics.csv'),'sku_policies':read('inventory_sku_metrics.csv'),'traces':read('inventory_trajectories.csv.gz'),'selected':read('selected_models.csv')}

d=load();s=d['summary'];champion=s['validation_champion']
names={r.StockCode:f'{r.StockCode} · {r.description}' for r in d['cohort'].itertuples()}
with st.sidebar:
    st.subheader('Explore a product')
    sku=st.selectbox('Product / SKU',list(names),format_func=names.get)
    view=st.selectbox('Forecast approach',['Validation champion','Per-SKU selection'])
    st.caption('UK customer-country sales · December 2009–December 2011 source · Historical analysis, not a forecast for today')
    st.caption('Zero weeks mean no recorded sales, not verified zero demand. Customer IDs are not used or included.')
selected_model=d['selected'].set_index('sku').loc[sku,'model']
model=champion if view=='Validation champion' else selected_model
test_metrics=pd.read_csv(ROOT/'outputs/test_metrics.csv')
score=test_metrics.loc[test_metrics.model.eq(champion)].iloc[0]
columns=st.columns(4)
columns[0].metric('Complete weeks',s['complete_weeks'])
columns[1].metric('Training-selected SKUs',s['selected_skus'])
columns[2].metric('Final test weeks',s['test_weeks'])
columns[3].metric('Champion test WAPE',f'{score.wape:.1%}')
st.caption(f'Validation champion: {champion}. WAPE is absolute forecast error divided by recorded sales volume; it is not an accuracy percentage. These headline metrics refer to the full 20-SKU cohort.')

def style(fig):
    fig.update_layout(template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'family':'Arial','color':'#183542'},margin={'l':20,'r':20,'t':35,'b':40},legend_title_text='')
    return fig
def download(frame):return frame.to_csv(index=False).encode('utf-8')

sales,evidence,inventory,methods=st.tabs(['Sales & forecasts','Model evidence','Stock policy lab','Methods & audit'])
with sales:
    st.subheader(names[sku])
    weekly=d['weekly'];fig=go.Figure(go.Scatter(x=weekly.week,y=weekly[sku],name='Recorded weekly sales',line={'color':'#167e83'}))
    fig.update_yaxes(title='Recorded units');fig.update_layout(height=330)
    st.plotly_chart(style(fig),width='stretch')
    st.subheader('Forecasts evaluated on later, unseen weeks')
    pred=d['predictions'].loc[d['predictions'].phase.eq('test') & d['predictions'].sku.eq(sku) & d['predictions'].model.eq(model)].sort_values('target_week').copy()
    buffer_policy=champion if view=='Validation champion' else 'Selected per SKU'
    bands=d['buffers'].loc[d['buffers'].policy.eq(buffer_policy)&d['buffers'].empirical_quantile.eq(.9)].set_index('protection_weeks').scaled_horizon_band.to_dict()
    widths=pred.horizon.map(bands)*s['series_training_means'][sku]
    pred['lower']=(pred.predicted-widths).clip(lower=0);pred['upper']=pred.predicted+widths
    chart=go.Figure()
    chart.add_trace(go.Scatter(x=pred.target_week,y=pred.upper,line={'width':0},showlegend=False))
    chart.add_trace(go.Scatter(x=pred.target_week,y=pred.lower,line={'width':0},fill='tonexty',fillcolor='rgba(22,126,131,.13)',name='Empirical calibration band'))
    chart.add_trace(go.Scatter(x=pred.target_week,y=pred.actual,name='Recorded sales',line={'color':'#132d3a','width':2}))
    chart.add_trace(go.Scatter(x=pred.target_week,y=pred.predicted,name=model,line={'color':'#bf792f','width':2,'dash':'dash'}))
    chart.update_yaxes(title='Units');st.plotly_chart(style(chart),width='stretch')
    st.caption('Four non-overlapping four-week test forecasts per SKU. The displayed bands use separately calibrated, horizon-specific pooled errors; a 90th-percentile calibration setting is not a guaranteed 90% prediction interval.')
    st.dataframe(pred[['origin_week','target_week','horizon','actual','predicted','lower','upper']],hide_index=True,width='stretch')
    st.download_button('Download selected backtest',download(pred),'sku_backtest.csv','text/csv')
    st.subheader('Demonstration forecast at the historical cut-off')
    future=d['future'].loc[d['future'].sku.eq(sku)&d['future'].policy.eq(view)]
    st.caption('Forecast origin: after the retained week ending 4 December 2011. These dates are historical; the source’s final partial week is excluded from fitting.')
    st.dataframe(future[['week','model','point','lower','upper']],hide_index=True,width='stretch')

with evidence:
    st.subheader('Simple benchmarks remain hard to beat')
    display=test_metrics.copy();display['WAPE (%)']=100*display.wape
    st.plotly_chart(style(px.bar(display,x='WAPE (%)',y='model',orientation='h',color_discrete_sequence=['#167e83'])),width='stretch')
    st.dataframe(test_metrics,hide_index=True,width='stretch')
    st.info('The global champion was chosen using the earlier selection period. Per-SKU selection has worse final-test error here. The best test result is reported for comparison, not used retrospectively to choose a new champion.')
    st.caption('MAE: mean absolute error in units. WAPE: volume-weighted aggregate absolute error. Bias ratio: total predicted-minus-recorded units / recorded units. MASE uses a lag-one scale estimated only on the initial 64 training weeks.')
    st.subheader('Selection evidence')
    st.dataframe(pd.read_csv(ROOT/'outputs/global_selection_scores.csv'),hide_index=True,width='stretch')
    st.dataframe(d['selected'][['sku','model','mae','wape']],hide_index=True,width='stretch')
    st.write(f"Four-week aggregate empirical-band test coverage for per-SKU forecasts: **{s['test_four_week_empirical_band_coverage']:.1%}**, across 80 SKU-origin windows. Correlated products and limited calendar origins restrict interpretation.")
    st.download_button('Download final-test metrics',download(test_metrics),'test_metrics.csv','text/csv')

with inventory:
    st.subheader('Inspect a replenishment assumption')
    st.warning('This is a hypothetical inventory trace replay driven by real recorded sales. Stock, supplier lead times and replenishment rules are assumptions. Unmet proxy units and replay fill are not the retailer’s actual lost sales or service levels.')
    a,b,c=st.columns(3)
    lead=a.selectbox('Assumed lead time (weeks)',[1,2,3])
    quantile=b.selectbox('Calibration buffer percentile',[.8,.9,.95],index=1,format_func=lambda x:f'{x:.0%}')
    policy=c.selectbox('Forecast policy',list(d['policies'].policy.unique()),index=list(d['policies'].policy.unique()).index(champion))
    scope=st.radio('Scenario scope',['All 20 SKUs','Selected SKU'],horizontal=True)
    source=d['policies'] if scope=='All 20 SKUs' else d['sku_policies'].loc[d['sku_policies'].sku.eq(sku)]
    result=source.loc[source.lead_weeks.eq(lead)&source.buffer_quantile.eq(quantile)&source.policy.eq(policy)].iloc[0]
    m=st.columns(3)
    m[0].metric('Replay fill',f'{result.proxy_fill_fraction:.1%}' if pd.notna(result.proxy_fill_fraction) else '—')
    m[1].metric('Mean ending units',f'{result.mean_ending_units_per_sku_week:,.1f}')
    m[2].metric('Unmet proxy units',f'{result.unmet_proxy_units:,.0f}')
    st.caption('Mean ending units are averaged across SKU-weeks. No sales driver means fill is undefined. A buffer percentile is an empirical error setting, not a guaranteed service level.')
    compare=source.loc[source.lead_weeks.eq(lead)].copy()
    st.plotly_chart(style(px.scatter(compare,x='mean_ending_units_per_sku_week',y='proxy_fill_fraction',color='policy',symbol='buffer_quantile',hover_data=['unmet_proxy_units'],labels={'mean_ending_units_per_sku_week':'Mean ending units per SKU-week','proxy_fill_fraction':'Replay fill fraction'},color_discrete_sequence=['#167e83','#bf792f','#7166a5'])),width='stretch')
    st.dataframe(compare,hide_index=True,width='stretch')
    st.caption('Retrospective policy comparison on the same historical trace and initialization. It is not an economic optimum: supplier constraints, costs, actual inventory and unobserved demand are absent.')
    trace=d['traces'].loc[d['traces'].sku.eq(sku)&d['traces'].lead_weeks.eq(lead)&d['traces'].buffer_quantile.eq(quantile)&d['traces'].policy.eq(policy)]
    st.subheader('Selected SKU stock-flow audit')
    st.dataframe(trace[['week','opening_after_arrival','arrivals','ordered','observed_sales_driver','served_proxy_units','unmet_proxy_units','ending_on_hand','ending_pipeline']],hide_index=True,width='stretch')
    st.download_button('Download selected stock-flow trace',download(trace),'inventory_trace.csv','text/csv')

with methods:
    st.subheader('Chronology before complexity')
    st.markdown('''**Data.** The UCI Online Retail II archive has two years of real transactions. UK customer-country lines with positive quantities and recorded prices, non-cancellation invoices and product-like codes are retained. Exact duplicates are counted but kept because the source does not prove they are mistakes. No customer ID is used. Partial first/last weeks are excluded; absent product-week sales are zero-filled as recorded-sales observations, not verified zero latent demand.

**Cohort.** Select the top 20 products by initial-training unit volume, among products active in at least 32 of 64 weeks and observed within the first eight retained weeks. Later sales or survival do not influence eligibility. This cohort covers 12.4% of initial-training retained units, not the full catalogue.

**Four blocks.** 64 initial training weeks; 12 selection weeks; 12 calibration weeks; 16 final-test weeks. Selection uses three non-overlapping four-week origins. Calibration uses three later origins; test uses four later origins. Each forecast uses strictly prior observations. Model choices remain fixed while fits update with newly observed history.

**Models.** Recent eight-week mean, 52-week seasonal naive, fixed-alpha exponential smoothing, Croston SBA, and recursively forecast ridge regression with past-only lag/rolling features plus known 52-week harmonics. Parameters are fixed in advance. Annual recurrence is weakly supported by only two years; wholesale bursts and product changes limit forecasts.

**Buffers.** Separate calibration errors are normalized by each SKU's initial-training mean and pooled across 20 SKUs × three origins. Positive protection-period errors give safety buffers; absolute errors give displayed empirical bands. Products and origins are dependent; neither distribution-free coverage nor future service performance is guaranteed.

**Inventory.** Review weekly before observing that week's sales driver. Protection horizon is lead time + one review week. Incoming orders arrive after the assumed lead time. Order quantity is the positive gap between an order-up-to target and on-hand-plus-pipeline stock. Initial stock is the ceiling of the prior eight-week mean × protection weeks, identical across forecasting policies for a given SKU and lead time. Shortages are treated as lost proxy units, with no backorders. Learning continues on the same historical sales trace, rather than simulated fulfilled units. Remaining stock and pipeline at the test end are reported; no terminal financial adjustment is applied.

**Limits.** Recorded sales are potentially censored by unknown stockouts. The source has no actual stock, lead times, purchase costs, supplier capacities or promotion data. No actual savings, economic optimum, uncensored-demand accuracy or current deployment benefit is claimed. SKU 84270 has zero recorded units in the final test and is retained, illustrating why future survival must not drive selection.''')
    st.dataframe(pd.read_csv(ROOT/'outputs/cleaning_audit.csv'),hide_index=True,width='stretch')
    st.markdown('[Source: Chen, D. (2012), Online Retail II, UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) · [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) · CC BY 4.0 · [Reproducible code](https://github.com/opeyemiagboolabethel/demand-forecasting-inventory-optimisation)')
    st.download_button('Download source manifest',(ROOT/'data/source_manifest.json').read_bytes(),'source_manifest.json','application/json')

st.caption('Prepared by Opeyemi Agboola · Forecasting, operational analytics and evidence-based decision support')
