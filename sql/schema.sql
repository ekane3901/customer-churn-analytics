-- sql/schema.sql
-- PostgreSQL schema for the customer churn analytics platform (AWS RDS)
-- Run this once to set up the database tables.

-- ============================================================
-- Raw customer data table (mirrors Telco CSV structure)
-- ============================================================
CREATE TABLE IF NOT EXISTS customers_raw (
    customer_id       VARCHAR(20) PRIMARY KEY,
    gender            VARCHAR(10),
    senior_citizen    SMALLINT,
    partner           VARCHAR(5),
    dependents        VARCHAR(5),
    tenure            INTEGER,
    phone_service     VARCHAR(5),
    multiple_lines    VARCHAR(25),
    internet_service  VARCHAR(25),
    online_security   VARCHAR(25),
    online_backup     VARCHAR(25),
    device_protection VARCHAR(25),
    tech_support      VARCHAR(25),
    streaming_tv      VARCHAR(25),
    streaming_movies  VARCHAR(25),
    contract          VARCHAR(20),
    paperless_billing VARCHAR(5),
    payment_method    VARCHAR(35),
    monthly_charges   NUMERIC(8, 2),
    total_charges     NUMERIC(10, 2),
    churn             SMALLINT,        -- 0 = No, 1 = Yes
    loaded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Engineered features table (populated by feature_queries.sql)
-- ============================================================
CREATE TABLE IF NOT EXISTS customer_features (
    customer_id          VARCHAR(20) PRIMARY KEY REFERENCES customers_raw(customer_id),
    tenure               INTEGER,
    monthly_charges      NUMERIC(8, 2),
    total_charges        NUMERIC(10, 2),
    clv_estimate         NUMERIC(12, 2),   -- monthly_charges * tenure
    services_count       SMALLINT,          -- number of add-on services subscribed
    is_month_to_month    SMALLINT,          -- 1 if contract = Month-to-month
    is_long_term         SMALLINT,          -- 1 if contract = One/Two year
    charge_per_service   NUMERIC(8, 2),     -- monthly_charges / services_count
    tenure_segment       VARCHAR(15),       -- new / developing / established / loyal
    churn                SMALLINT,
    engineered_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Model predictions table
-- ============================================================
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id         SERIAL PRIMARY KEY,
    customer_id           VARCHAR(20) REFERENCES customers_raw(customer_id),
    model_name            VARCHAR(50),
    churn_probability     NUMERIC(6, 4),
    predicted_churn       SMALLINT,          -- 0 or 1 based on threshold
    clv_estimate          NUMERIC(12, 2),
    revenue_risk_score    NUMERIC(14, 2),    -- churn_probability * clv_estimate
    predicted_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast risk score lookups
CREATE INDEX IF NOT EXISTS idx_risk_score
    ON churn_predictions (revenue_risk_score DESC);

CREATE INDEX IF NOT EXISTS idx_customer_predictions
    ON churn_predictions (customer_id, predicted_at DESC);
