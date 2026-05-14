-- sql/risk_scores.sql
-- Business reporting queries for churn predictions.
-- These are the queries that power the dashboard and executive summaries.


-- ============================================================
-- 1. Top 25 customers by revenue risk score
--    (Who should the retention team call first?)
-- ============================================================
SELECT
    p.customer_id,
    p.churn_probability,
    p.clv_estimate,
    p.revenue_risk_score,
    r.contract,
    r.tenure,
    r.monthly_charges,
    r.internet_service,
    r.payment_method
FROM churn_predictions p
JOIN customers_raw r USING (customer_id)
WHERE p.model_name = 'xgboost'
  AND p.predicted_churn = 1
ORDER BY p.revenue_risk_score DESC
LIMIT 25;


-- ============================================================
-- 2. Total revenue at risk by customer segment
-- ============================================================
SELECT
    f.tenure_segment,
    COUNT(*)                                     AS customer_count,
    ROUND(AVG(p.churn_probability), 3)           AS avg_churn_prob,
    ROUND(SUM(p.revenue_risk_score), 2)          AS total_revenue_at_risk,
    ROUND(AVG(f.monthly_charges), 2)             AS avg_monthly_charges
FROM churn_predictions p
JOIN customer_features f USING (customer_id)
WHERE p.model_name = 'xgboost'
GROUP BY f.tenure_segment
ORDER BY total_revenue_at_risk DESC;


-- ============================================================
-- 3. Churn rate and revenue risk by contract type
-- ============================================================
SELECT
    r.contract,
    COUNT(*)                                     AS total_customers,
    SUM(r.churn)                                 AS actual_churned,
    ROUND(100.0 * SUM(r.churn) / COUNT(*), 1)   AS churn_rate_pct,
    ROUND(AVG(p.churn_probability), 3)           AS avg_predicted_churn_prob,
    ROUND(SUM(p.revenue_risk_score), 2)          AS total_revenue_at_risk
FROM customers_raw r
LEFT JOIN churn_predictions p USING (customer_id)
WHERE p.model_name = 'xgboost'
GROUP BY r.contract
ORDER BY churn_rate_pct DESC;


-- ============================================================
-- 4. Monthly revenue at risk summary (dashboard KPI cards)
-- ============================================================
SELECT
    COUNT(*)                                                AS total_customers,
    SUM(CASE WHEN predicted_churn = 1 THEN 1 ELSE 0 END)  AS predicted_churners,
    ROUND(
        100.0 * SUM(CASE WHEN predicted_churn = 1 THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                       AS predicted_churn_rate_pct,
    ROUND(SUM(revenue_risk_score), 2)                       AS total_revenue_at_risk,
    ROUND(AVG(churn_probability), 4)                        AS avg_churn_probability
FROM churn_predictions
WHERE model_name = 'xgboost';
