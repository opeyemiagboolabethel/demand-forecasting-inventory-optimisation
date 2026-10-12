# Demand Forecasting & Inventory Optimisation

A commercial data-science project by Opeyemi Agboola. It combines real retail sales, chronological forecasting evaluation and an explicit replenishment-policy laboratory. The goal is to inspect forecasts and inventory trade-offs without inventing operational facts or claiming savings the source cannot establish.

## Source and scope

Source: Chen, D. (2012), **Online Retail II**, UCI Machine Learning Repository, [DOI 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), [source page](https://archive.ics.uci.edu/dataset/502/online+retail+ii). UCI describes two years of real transactions from a UK online giftware retailer, with many wholesale customers. Dataset licence: **CC BY 4.0**. Attribution and this licence apply to derived data.

The archive contains **1,067,371 transaction lines**. After the documented sequential filters, **922,753** UK customer-country, positive-quantity/price, non-cancellation product lines fall inside 104 complete Monday–Sunday weeks. Customer IDs are not read or included. Exact duplicate lines are counted but retained because the source does not prove they are mistakes.

The 20-SKU cohort is selected entirely from the first 64 weeks, among products active in at least 32 weeks and observed within the first eight. These products represent **12.37%** of initial-training retained units, not the full catalogue. Future activity never determines eligibility; SKU 84270 remains in the analysis despite having zero final-test sales.

## What the evidence actually shows

The global selection-period champion is the **recent eight-week mean**. Final-test WAPE is **60.16%**; per-SKU model selection produces **78.64%**. Exponential smoothing has the lowest observed test error, **59.70%**, but is not retrospectively substituted as the champion. Complexity does not automatically improve this cohort's forecasts.

Recorded sales are only a demand proxy. Actual stock levels, stockouts, supplier lead times, purchase costs and promotion data are absent. Inventory results are **hypothetical trace-replay scenarios**, driven by real observed sales—not the retailer's actual inventory performance. No actual savings, profit optimum or uncensored-demand accuracy is claimed.

## Run the application

Use **Python 3.12**. Committed processed outputs let the application run without the raw workbook, API keys, credentials or live data requests.

```sh
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Streamlit Community Cloud: repository `opeyemiagboolabethel/demand-forecasting-inventory-optimisation`, branch `main`, entrypoint `streamlit_app.py`, Advanced settings → Python **3.12**. No secrets required.

## Reproduce

```sh
pip install -r requirements-analysis.txt
python download_data.py
python prepare_data.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m src.pipeline
python -m unittest test_project.py
python make_figures.py
```

The downloader verifies the archive against its recorded SHA-256 hash. Historical upstream changes cause an explicit failure. The raw 45.6 MB archive and regenerated SQLite database are excluded from Git; the committed cohort, weekly observations, predictions and replay outputs support the app and evidence tests. Excel processing takes longer than running the app.

## Four chronological blocks

| Block | Complete weeks | Purpose |
| --- | ---: | --- |
| Initial training | 64 | SKU eligibility, initial scaling and first model fit |
| Model selection | 12 | Three non-overlapping four-week forecast origins; freeze champion/per-SKU choices |
| Error calibration | 12 | Three later origins; estimate empirical buffers without the final test |
| Final test | 16 | Four later origins; report performance and replay 27 policy scenarios |

Retained history begins 7 December 2009 and ends with the week beginning 28 November 2011. The test begins 15 August 2011. The first and last partial source weeks are omitted. Forecasts after 4 December 2011 are demonstration forecasts at the historical cut-off, not forecasts for the present.

Each origin fits only observations before that origin. Model choices stay fixed; fits update as new history becomes available. No random train/test split is used. Product descriptions are taken from initial-training data.

## Forecast candidates

- Recent mean over eight weeks.
- Seasonal naive: same weekly position 52 weeks earlier.
- Exponential smoothing with alpha 0.2.
- Croston SBA with alpha 0.1, including the 0.95 bias adjustment.
- Ridge regression with alpha 10, standardized past-only lag/rolling features and known 52-week harmonics; future horizons are forecast recursively.

Parameters are fixed before selection. Two years provide limited annual-cycle evidence. Forecasts are constrained to be nonnegative. MAE, aggregate WAPE, bias ratio and lag-one MASE are reported. MASE scales use only the initial 64 weeks.

## Calibration and scenario mechanics

Calibration errors are scaled by each SKU's initial-training mean and pooled across 20 SKUs × three origins. Horizon-specific absolute errors create displayed bands. Positive aggregate errors over lead + one review week create safety buffers. Quantiles use the observed higher-order statistic. These are empirical settings; dependent products and origins prevent a distribution-free coverage claim or guaranteed service-level interpretation.

Inventory is reviewed at the start of each week, before observing that week's sales driver. Outstanding orders arrive after an assumed one-, two- or three-week lead time. The order-up-to target is the ceiling of the protection-period forecast plus the calibrated buffer. Order quantity is the positive gap between that target and on-hand-plus-pipeline stock.

Initial stock equals the ceiling of the previous eight-week mean × protection weeks, with no initial pipeline. It is identical across policies for a given SKU and lead time. Shortfalls are treated as lost proxy units, not backorders. Every policy learns from the same historical sales driver, even if a replay policy cannot fulfil it. Remaining stock and pipeline are retained at the test end; no terminal financial adjustment is invented.

The 27 scenarios combine three forecast policies, three lead times and three calibration percentiles. Replay fill and average ending inventory expose trade-offs. They do not establish an economic optimum because costs and constraints are unknown.

## Files

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Source cleaning, audit, training-only cohort and complete weekly observations |
| `src/forecasting.py` | Forecasting functions and stock-flow mechanics |
| `src/pipeline.py` | Selection, separate calibration, final test and scenario generation |
| `analysis.sql` | Executed global/SKU inventory summaries |
| `streamlit_app.py` | Forecast explorer, model evidence and stock-policy laboratory |
| `outputs/selection_scores.csv` | All per-SKU selection scores |
| `outputs/global_selection_scores.csv` | Global champion selection evidence |
| `outputs/all_rolling_predictions.csv.gz` | Every evaluated origin/horizon/model forecast |
| `outputs/calibration_errors.csv` | Pre-test error evidence |
| `outputs/inventory_trajectories.csv.gz` | Auditable stock-flow traces for all scenarios |
| `test_project.py` | Eight evidence and interaction checks |

## Limits

Wholesale bursts, product changes and unknown promotions create large errors. Zero recorded sales may reflect genuine inactivity, stockouts or recording/availability gaps. Cancellations and negative quantities are excluded rather than treated as observed stock returns. Duplicate treatment is an explicit source-quality uncertainty; no deduplication-sensitivity result is claimed.

The selected cohort favours established, active products. It is not evidence about new launches, the whole retailer, current operations or Nigerian retail. Band coverage and policy comparisons are historical, conditional results. No causal or production-benefit claim is made.
