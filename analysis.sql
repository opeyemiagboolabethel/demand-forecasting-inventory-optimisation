-- Reported units are real observed sales; stock, lead times and policies are hypothetical scenarios.
-- report: inventory_policy_metrics
SELECT policy,lead_weeks,buffer_quantile,
 SUM(observed_sales_driver) AS observed_sales_driver_units,
 SUM(served_proxy_units) AS served_proxy_units,
 SUM(unmet_proxy_units) AS unmet_proxy_units,
 SUM(served_proxy_units)/SUM(observed_sales_driver) AS proxy_fill_fraction,
 AVG(ending_on_hand) AS mean_ending_units_per_sku_week,
 SUM(ordered) AS ordered_units,
 SUM(CASE WHEN ordered>0 THEN 1 ELSE 0 END) AS order_events
FROM inventory_trace GROUP BY policy,lead_weeks,buffer_quantile
ORDER BY lead_weeks,buffer_quantile,policy;

-- report: inventory_sku_metrics
SELECT policy,sku,lead_weeks,buffer_quantile,
 SUM(observed_sales_driver) AS observed_sales_driver_units,
 SUM(served_proxy_units)/SUM(observed_sales_driver) AS proxy_fill_fraction,
 AVG(ending_on_hand) AS mean_ending_units_per_sku_week,
 SUM(unmet_proxy_units) AS unmet_proxy_units
FROM inventory_trace GROUP BY policy,sku,lead_weeks,buffer_quantile
ORDER BY sku,lead_weeks,buffer_quantile,policy;
