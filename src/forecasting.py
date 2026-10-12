"""Fixed forecasting candidates. Every fit sees observations strictly before its origin."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS=['Recent mean (8 weeks)','Seasonal naive (52 weeks)','Exponential smoothing (alpha .2)','Croston SBA (alpha .1)','Ridge lag regression']

def lag_features(history, week_index):
    y=np.asarray(history,dtype=float)
    return [y[-1],y[-2],y[-4],y[-8],y[-4:].mean(),y[-13:].mean(),np.sin(2*np.pi*week_index/52),np.cos(2*np.pi*week_index/52)]

def forecast(history, horizon, model):
    y=np.asarray(history,dtype=float)
    if len(y)<52 or horizon<1:raise ValueError('Requires at least 52 weeks and a positive horizon')
    if model==MODELS[0]:pred=np.full(horizon,y[-8:].mean())
    elif model==MODELS[1]:
        values=list(y)
        for _ in range(horizon):values.append(values[-52])
        pred=np.array(values[-horizon:])
    elif model==MODELS[2]:
        level=y[0]
        for v in y[1:]:level=.2*v+.8*level
        pred=np.full(horizon,level)
    elif model==MODELS[3]:
        nz=np.flatnonzero(y>0)
        if not len(nz):pred=np.zeros(horizon)
        else:
            size=y[nz[0]];interval=float(nz[0]+1);previous=int(nz[0])
            for ix in nz[1:]:size=.1*y[ix]+.9*size;interval=.1*(ix-previous)+.9*interval;previous=int(ix)
            pred=np.full(horizon,.95*size/max(interval,1))
    elif model==MODELS[4]:
        features=np.array([lag_features(y[:t],t) for t in range(13,len(y))])
        fitted=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(features,y[13:])
        values=list(y);pred=[]
        for h in range(horizon):
            v=max(0,float(fitted.predict([lag_features(values,len(values))])[0]));values.append(v);pred.append(v)
        pred=np.array(pred)
    else:raise ValueError('Unknown forecasting model')
    return np.maximum(0,pred)

def simulate(trace, future_predictions, buffer_units, lead_weeks, start, stop, initial_stock):
    """Lost-sales trace replay under explicit hypothetical weekly-review assumptions."""
    if lead_weeks<1 or initial_stock<0:raise ValueError('Invalid scenario')
    hand=float(initial_stock);orders={};rows=[]
    for t in range(start,stop):
        arrival=orders.pop(t,0.0);hand+=arrival
        pipeline=sum(orders.values())
        target=float(np.ceil(sum(future_predictions[t][:lead_weeks+1])+buffer_units))
        ordered=max(0.0,target-hand-pipeline)
        orders[t+lead_weeks]=orders.get(t+lead_weeks,0.0)+ordered
        observed=float(trace[t]);served=min(hand,observed);unmet=observed-served;hand-=served
        rows.append({'week_index':t,'opening_after_arrival':hand+served,'arrivals':arrival,'order_up_to':target,'ordered':ordered,'observed_sales_driver':observed,'served_proxy_units':served,'unmet_proxy_units':unmet,'ending_on_hand':hand,'ending_pipeline':sum(orders.values())})
    return rows
