-- sql/feature_queries.sql
-- SQL-based feature engineering on AWS RDS (PostgreSQL)
-- Populates the customer_features table from customers_raw.
-- These queries demonstrate SQL skills for feature creation.


-- ============================================================
-- Helper: Count of add-on services per customer
-- ============================================================
-- Each service column contains 'Yes', 'No', or 'No internet/phone service'
-- We count only actual 'Yes' values.

WITH service_counts AS (
    SELECT
        customer_id,
        (
            (CASE WHEN phone_service    = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN multiple_lines   = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN online_security  = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN online_backup    = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN device_protection= 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN tech_support     = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN streaming_tv     = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END)
        ) AS services_count
    FROM customers_raw
),


-- ============================================================
-- Helper: CLV estimate and derived features
-- ============================================================
clv_calc AS (
    SELECT
        r.customer_id,
        r.tenure,
        r.monthly_charges,
        r.total_charges,
        r.monthly_charges * r.tenure                              AS clv_estimate,
        CASE WHEN r.contract = 'Month-to-month' THEN 1 ELSE 0 END AS is_month_to_month,
        CASE WHEN r.contract IN ('One year', 'Two year') THEN 1 ELSE 0 END AS is_long_term,
        CASE
            WHEN r.tenure BETWEEN 0  AND 12 THEN 'new'
            WHEN r.tenure BETWEEN 13 AND 24 THEN 'developing'
            WHEN r.tenure BETWEEN 25 AND 48 THEN 'established'
            ELSE 'loyal'
        END AS tenure_segment,
        r.churn
    FROM customers_raw r
)


-- ============================================================
-- Final INSERT into customer_features
-- ============================================================
INSERT INTO customer_features (
    customer_id,
    tenure,
    monthly_charges,
    total_charges,
    clv_estimate,
    services_count,
    is_month_to_month,
    is_long_term,
    charge_per_service,
    tenure_segment,
    churn
)
SELECT
    c.customer_id,
    c.tenure,
    c.monthly_charges,
    c.total_charges,
    c.clv_estimate,
    s.services_count,
    c.is_month_to_month,
    c.is_long_term,
    -- Avoid division by zero for customers with no services
    CASE
        WHEN s.services_count = 0 THEN c.monthly_charges
        ELSE ROUND(c.monthly_charges / s.services_count, 2)
    END AS charge_per_service,
    c.tenure_segment,
    c.churn
FROM clv_calc c
JOIN service_counts s USING (customer_id)
ON CONFLICT (customer_id) DO UPDATE
    SET
        clv_estimate       = EXCLUDED.clv_estimate,
        services_count     = EXCLUDED.services_count,
        charge_per_service = EXCLUDED.charge_per_service,
        engineered_at      = CURRENT_TIMESTAMP;
