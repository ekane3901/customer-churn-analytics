# Customer Churn & Revenue Risk Analytics Platform

> An end-to-end machine learning system to predict customer churn and quantify revenue risk — built with Python, SQL, and AWS.

---

## Project Overview

Customer churn is one of the most costly problems in subscription-based businesses. This platform ingests multi-source customer data, engineers predictive features via SQL, trains classification models, and surfaces high-risk customers through an interactive dashboard — framed as a **revenue risk score**, not just a churn probability.

**Business question answered:** *Which customers are most likely to leave, and how much revenue is at risk if they do?*

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
AWS S3 (data lake)
     │
     ▼
AWS RDS / PostgreSQL (SQL feature engineering)
     │
     ▼
Python ML Pipeline (EDA → Feature Engineering → Modeling)
     │
     ▼
Trained Model + Risk Scores
     │
     ▼
Streamlit Dashboard (churn risk + revenue impact)
```

---

## Tech Stack

| Layer            | Tool/Service                        |
|------------------|-------------------------------------|
| Cloud Storage    | AWS S3                              |
| Database         | AWS RDS (PostgreSQL)                |
| Compute          | AWS EC2 / Local                     |
| Data Processing  | Python, Pandas, NumPy               |
| SQL Engineering  | PostgreSQL, SQLAlchemy              |
| Modeling         | Scikit-learn, XGBoost               |
| Visualization    | Matplotlib, Seaborn, Plotly         |
| Dashboard        | Streamlit                           |
| Version Control  | Git / GitHub                        |
| Environment      | Python 3.11+, virtualenv            |

---

## Dataset

**Source:** [Telco Customer Churn — IBM Sample Dataset via Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Key fields:**
- `customerID` — unique customer identifier
- `tenure` — months as a customer
- `MonthlyCharges`, `TotalCharges` — billing features
- `Contract` — month-to-month, one year, two year
- `InternetService`, `TechSupport`, `StreamingTV` — service features
- `Churn` — target variable (Yes/No)

**Why this dataset:** Real-world structure, class imbalance (~27% churn rate), mix of categorical and numerical features — mirrors what you'd encounter in industry.

---

## Project Structure

```
customer-churn-analytics/
│
├── data/
│   ├── raw/                    # Original unmodified data (gitignored)
│   └── processed/              # Cleaned, feature-engineered data (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb       # Model training and evaluation
│   └── 04_risk_scoring.ipynb   # Revenue risk scoring logic
│
├── src/
│   ├── data/
│   │   ├── ingest.py           # Load data from S3 / local
│   │   └── preprocess.py       # Cleaning and type casting
│   ├── features/
│   │   └── engineer.py         # Feature engineering functions
│   ├── models/
│   │   ├── train.py            # Model training pipeline
│   │   ├── evaluate.py         # Metrics and evaluation
│   │   └── predict.py          # Inference / scoring
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
│
├── sql/
│   ├── schema.sql              # RDS table definitions
│   ├── feature_queries.sql     # SQL-based feature engineering
│   └── risk_scores.sql         # Revenue risk aggregations
│
├── models/                     # Saved model artifacts (gitignored)
│
├── reports/
│   └── figures/                # EDA plots, model evaluation charts
│
├── config/
│   └── config.yaml             # Project config (paths, model params)
│
├── tests/
│   └── test_features.py        # Unit tests for feature engineering
│
├── requirements.txt
├── setup.py
├── .env.example                # Template for environment variables
├── .gitignore
└── README.md
```

---

## Modeling Approach

### Models Trained
1. **Logistic Regression** — interpretable baseline; explains which features drive churn
2. **XGBoost Classifier** — primary model; handles non-linearity and feature interactions

### Evaluation Metrics
| Metric | Why It Matters |
|--------|----------------|
| AUC-ROC | Overall discrimination; robust to class imbalance |
| Precision-Recall | Focus on catching churners without too many false alarms |
| F1 Score | Balanced recall/precision tradeoff |
| Log Loss | Calibration of probabilities |

> Accuracy is intentionally *not* the primary metric — a model predicting "no churn" always would hit ~73% accuracy on this dataset, which is misleading.

### Revenue Risk Score
```
Risk Score = Churn Probability × Estimated Customer Lifetime Value (CLV)
```
This ranks customers not just by likelihood to churn, but by *how much it costs the business* if they do.

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/customer-churn-analytics.git
cd customer-churn-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Fill in your AWS credentials and RDS connection string
```

### 5. Download dataset
Place the Telco CSV from Kaggle into `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Running the Project

```bash
# Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# Train models
python src/models/train.py

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Key Findings *(updated as project progresses)*

- [ ] Top features driving churn (EDA phase)
- [ ] Model AUC-ROC scores
- [ ] Revenue at risk from top 20% predicted churners
- [ ] Segment with highest churn rate

---

## Status

| Phase | Status |
|-------|--------|
| Data ingestion & EDA | 🔄 In Progress |
| SQL feature engineering | ⏳ Planned |
| Model training | ⏳ Planned |
| Revenue risk scoring | ⏳ Planned |
| Streamlit dashboard | ⏳ Planned |
| AWS deployment | ⏳ Planned |

---

## Author

**Your Name** — [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

*Built as a portfolio project to demonstrate end-to-end data science skills: SQL, Python, ML modeling, cloud infrastructure, and business analytics.*
