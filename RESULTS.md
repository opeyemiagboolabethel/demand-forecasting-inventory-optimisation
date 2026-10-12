# Findings and decision implications

## A chronological benchmark, not a hand-picked success story

Twenty products are selected from initial-training data and evaluated over 104 complete weeks. The four blocks separate initial training, model selection, error calibration and final testing. Later product survival does not influence the cohort; one selected SKU has zero final-test sales and remains included.

| Approach | Final-test WAPE | MAE (units) | Interpretation |
| --- | ---: | ---: | --- |
| Recent eight-week mean — validation champion | 60.16% | 277.03 | Chosen before calibration/test; credible simple reference |
| Exponential smoothing | 59.70% | 274.92 | Lowest observed test error; not substituted retrospectively |
| Croston SBA | 60.37% | 277.99 | Similar aggregate error to the simple references |
| Ridge lag regression | 75.23% | 346.43 | More complex recursion does not improve this cohort |
| Per-SKU model selection | 78.64% | 362.13 | Local selection fails to generalize as well as the global champion |
| 52-week seasonal naive | 126.14% | 580.89 | Prior-year recurrence is a poor assumption here |

WAPE is an error ratio, not an accuracy percentage. The initial cohort accounts for 12.37% of retained training units. Results do not generalize to the full catalogue or current retail.

## Calibration needs its own evidence

The separate calibration period provides 60 SKU-origin windows per protection horizon for each policy. Four-week aggregate empirical bands for the per-SKU approach cover **86.25%** of the final-test windows, despite using a 90th-percentile calibration setting. This is observed historical coverage, not a guarantee; products and origins are dependent, and only three calibration calendar origins are available.

## Inventory trade-offs are visible, but hypothetical

At a **one-week assumed lead time** and **90th-percentile positive-error buffer**, the common test trace contains 147,359 recorded units across the 20 products.

| Forecast policy | Replay fill | Mean ending units per SKU-week | Unmet proxy units |
| --- | ---: | ---: | ---: |
| Recent eight-week mean | 85.81% | 623.63 | 20,910 |
| Per-SKU selection | 87.47% | 919.75 | 18,467 |
| 52-week seasonal naive | 92.72% | 1,760.29 | 10,731 |

The prior-year policy fulfils more of this fixed sales driver while carrying substantially more stock. Its overforecasting is consistent with that trade-off; a better fill figure alone does not establish a better economic policy. The source has no purchase/holding/shortage costs or supplier constraints with which to price the trade-off.

These are simulated inventory mechanics driven by real sales. The lead times, starting stock, buffer rule and lost-sales treatment are explicit assumptions. Replay shortages are not actual retailer stockouts or verified lost demand. No financial savings or deployable optimum is claimed.

## What a business should investigate next

Before implementation, obtain actual inventory position, replenishment lead-time distributions, supplier constraints, promotion information, unit costs and stockout records. Evaluate the policy prospectively using the required service/cost trade-off. Check discontinued and new products separately rather than removing inconvenient zero-sales series after seeing the test.

The project supports auditable forecasting and scenario review. Stronger accuracy and implementation claims require additional data and validation.

## Source

Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Dataset and derived data: CC BY 4.0. The company is a historical UK giftware retailer, with many wholesalers.
