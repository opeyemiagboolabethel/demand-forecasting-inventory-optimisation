"""Render exact forecasting and stock-trade-off figures from committed results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':11})
names={'Recent mean (8 weeks)':'Eight-week mean (validation champion)','Exponential smoothing (alpha .2)':'Exponential smoothing','Croston SBA (alpha .1)':'Croston SBA','Ridge lag regression':'Ridge lag regression','Selected per SKU':'Per-SKU model selection','Seasonal naive (52 weeks)':'52-week seasonal naive'}
scores=pd.read_csv(ROOT/'outputs/test_metrics.csv').sort_values('wape',ascending=False)
fig,ax=plt.subplots(figsize=(11,5.5));fig.patch.set_facecolor('#f7f7f2');ax.set_facecolor('#f7f7f2')
ax.barh(scores.model.map(names),scores.wape*100,color=['#bf792f' if m=='Selected per SKU' else '#167e83' for m in scores.model],height=.6)
for y,row in enumerate(scores.itertuples()):ax.text(row.wape*100+1.7,y,f'{row.wape:.1%}',va='center',fontsize=10)
ax.set_xlim(0,145);ax.set_xlabel('Final-test WAPE (%) · lower is better');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for spine in ax.spines.values():spine.set_visible(False)
ax.tick_params(length=0)
fig.suptitle('Complexity did not improve this sales forecast.',fontsize=18,fontweight='bold',x=.02,ha='left',color='#132d3a')
fig.text(.02,.02,'20 training-selected SKUs · 16 final-test weeks · WAPE is an error ratio, not accuracy.\nThe champion was chosen in the earlier selection period; the test did not choose it.',fontsize=10,color='#526b75')
fig.tight_layout(rect=[0,.10,1,.93]);fig.savefig(ROOT/'outputs/forecast-comparison.svg');fig.savefig('/tmp/demand-forecast-comparison.png',dpi=120);plt.close(fig)
policy=pd.read_csv(ROOT/'outputs/inventory_policy_metrics.csv');policy=policy.loc[policy.lead_weeks.eq(1)]
fig,ax=plt.subplots(figsize=(11,5.5));fig.patch.set_facecolor('#f7f7f2');ax.set_facecolor('#f7f7f2')
for (name,group),color in zip(policy.groupby('policy'),['#167e83','#7166a5','#bf792f']):
    group=group.sort_values('buffer_quantile');ax.plot(group.mean_ending_units_per_sku_week,group.proxy_fill_fraction*100,'o-',color=color,label=names.get(name,name),linewidth=2,markersize=7)
    for row in group.itertuples():ax.annotate(f'{row.buffer_quantile:.0%}',(row.mean_ending_units_per_sku_week,row.proxy_fill_fraction*100),xytext=(6,5),textcoords='offset points',fontsize=9)
ax.set_xlabel('Mean ending units per SKU-week');ax.set_ylabel('Replay fill (%)');ax.set_ylim(0,100);ax.set_xlim(left=0);ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('A higher replay fill can require much more stock.',fontsize=18,fontweight='bold',x=.02,ha='left',color='#132d3a')
fig.text(.02,.02,'One-week assumed lead time · Labels are calibration buffer percentiles.\nHypothetical stock mechanics driven by real sales; no actual inventory or savings claim.',fontsize=10,color='#526b75')
fig.tight_layout(rect=[0,.10,1,.93]);fig.savefig(ROOT/'outputs/inventory-tradeoff.svg');fig.savefig('/tmp/demand-inventory-tradeoff.png',dpi=120);plt.close(fig)
