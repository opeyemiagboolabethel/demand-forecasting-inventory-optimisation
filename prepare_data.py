"""Aggregate real sales; choose the SKU cohort using initial-training data only."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent

def main():
    raw = ROOT / 'data/raw/online_retail_ii.zip'
    source = json.loads((ROOT / 'data/source_manifest.json').read_text())
    if hashlib.sha256(raw.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Source checksum mismatch')
    with zipfile.ZipFile(raw) as archive:
        workbook = pd.ExcelFile(io.BytesIO(archive.read('online_retail_II.xlsx')))
        frames = []
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet, dtype={'Invoice': str, 'StockCode': str}, usecols=['Invoice','StockCode','Description','Quantity','InvoiceDate','Price','Country'])
            frames.append(frame)
            print(sheet, len(frame), flush=True)
    data = pd.concat(frames, ignore_index=True)
    audit = [{'stage':'Source transaction lines','rows':len(data)}]
    duplicate_lines = int(data.duplicated().sum())
    data.InvoiceDate = pd.to_datetime(data.InvoiceDate, errors='coerce')
    first, last = data.InvoiceDate.min().normalize(), data.InvoiceDate.max().normalize()
    first_week = first + pd.Timedelta(days=(7-first.weekday()) % 7)
    final_week = last - pd.Timedelta(days=last.weekday()) - pd.Timedelta(weeks=1)
    weeks = pd.date_range(first_week, final_week, freq='W-MON')
    if len(weeks) != 104:
        raise ValueError('Unexpected source coverage; fixed split requires 104 complete weeks')
    filters = [('United Kingdom customer-country rows', data.Country.eq('United Kingdom'))]
    data = data.loc[filters[0][1]].copy(); audit.append({'stage':filters[0][0],'rows':len(data)})
    for name, rule in [
      ('Valid timestamps', lambda d:d.InvoiceDate.notna()),
      ('Not cancellation invoice', lambda d:~d.Invoice.str.upper().str.startswith('C', na=False)),
      ('Positive quantity', lambda d:d.Quantity.gt(0)),
      ('Positive recorded price', lambda d:d.Price.gt(0)),
      ('Product-code format: five digits, up to two letters', lambda d:d.StockCode.str.fullmatch(r'\d{5}[A-Z]{0,2}',na=False)),
      ('Complete Monday–Sunday source weeks', lambda d:d.InvoiceDate.ge(first_week) & d.InvoiceDate.lt(final_week+pd.Timedelta(weeks=1))),
    ]:
        data = data.loc[rule(data)].copy();audit.append({'stage':name,'rows':len(data)})
    data['week'] = data.InvoiceDate.dt.normalize() - pd.to_timedelta(data.InvoiceDate.dt.weekday, unit='D')
    training = data.loc[data.week.lt(weeks[64])]
    grouped = training.groupby('StockCode').agg(training_units=('Quantity','sum'),active_training_weeks=('week','nunique'),first_sale_week=('week','min'))
    eligible = grouped.loc[grouped.active_training_weeks.ge(32) & grouped.first_sale_week.le(weeks[7])].sort_values(['training_units','StockCode'],ascending=[False,True])
    selected = eligible.head(20).copy()
    descriptions = training.loc[training.StockCode.isin(selected.index)].groupby('StockCode').Description.agg(lambda s:s.dropna().mode().iloc[0] if len(s.dropna()) else 'Description unavailable')
    selected['description'] = descriptions
    selected.reset_index().to_csv(ROOT / 'outputs/sku_cohort.csv',index=False)
    weekly = data.loc[data.StockCode.isin(selected.index)].groupby(['StockCode','week']).Quantity.sum().unstack('StockCode').reindex(weeks).fillna(0)
    weekly.index.name='week';weekly.astype('int64').to_csv(ROOT / 'outputs/weekly_sales.csv')
    pd.DataFrame(audit).to_csv(ROOT / 'outputs/cleaning_audit.csv',index=False)
    metadata={'source_rows':audit[0]['rows'],'retained_rows':len(data),'exact_duplicate_source_lines_retained':duplicate_lines,'complete_weeks':len(weeks),'start_week':str(weeks[0].date()),'last_week':str(weeks[-1].date()),'selected_skus':len(selected),'eligible_training_skus':len(eligible),'initial_train_weeks':64,'selection_weeks':12,'calibration_weeks':12,'test_weeks':16,'selection_start':str(weeks[64].date()),'calibration_start':str(weeks[76].date()),'test_start':str(weeks[88].date()),'first_omitted_partial_start':str(first.date()),'last_omitted_partial_start':str((final_week+pd.Timedelta(weeks=1)).date()),'selected_training_unit_share':float(selected.training_units.sum()/training.Quantity.sum()),'missing_customer_ids_policy':'Customer ID is not read or used; anonymous valid sales remain included.','duplicate_policy':'Exact duplicate lines are counted but retained; the source does not prove they are errors.'}
    (ROOT/'outputs/data_summary.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)

if __name__ == '__main__':main()
