# Retail Sales Analysis and Prediction Using Machine Learning

An end-to-end enterprise Machine Learning and Business Intelligence solution for multi-horizon store sales forecasting, built upon the **Kaggle Store Sales – Time Series Forecasting** dataset (Corporación Favorita, Ecuador).

The platform pairs rigorous, leak-free time-series feature engineering and gradient boosting with a multi-page interactive **Streamlit** dashboard and automated **ReportLab** PDF report generation.

---

## 🚀 Key Highlights & Business Capabilities

- **Zero-Leakage Feature Engineering**: Shifted historical lags (Lag 1, 7, 14, 30) and shifted rolling windows (Mean 7, 14, 30; Std 7) computed group-wise per `(store_nbr, family)`.
- **Chronological Model Benchmark**: Evaluates Seasonal Naive Baseline, Random Forest Regressor, and XGBoost Regressor on time-ordered validation and test holdout splits (August 2017).
- **Champion Model Selection**: XGBoost Regressor achieves an out-of-sample Test **RMSLE of 0.3842**, **WAPE of 0.1409**, and **R² of 0.9659**.
- **Recursive Forecasting Engine**: Multi-step recursive forecasting for **Daily (Next Day)**, **Weekly (7 Days)**, and **Monthly (30 Days)** horizons with dynamic lag updating and promotional sensitivity simulation.
- **Interactive Multi-Page Streamlit Dashboard**: 6 specialized analytical views equipped with Plotly visualizations, KPI cards, diagnostic residual plots, and export capabilities.
- **Automated PDF & CSV Reporting**: Instant generation of executive briefing dossiers via ReportLab and tabular CSV extracts.

---

## 🏗️ Project Architecture & Directory Layout

```
Retail-Sales-ML/
│
├── config.py                       # Central paths, feature lists, hyperparams, date ranges
├── requirements.txt                # Pinned dependencies
├── README.md                       # Comprehensive project documentation
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── raw/                        # Kaggle raw CSV files (train, test, stores, oil, holidays, tx)
│   └── processed/                  # Cleaned and engineered Parquet datasets
│       ├── cleaned_sales.parquet
│       ├── feature_matrix.parquet
│       └── test_evaluation_results.parquet
│
├── notebooks/
│   ├── 01_data_understanding.ipynb # Schema inspection, missingness, cardinality
│   ├── 02_preprocessing.ipynb      # Holiday dedup, oil interpolation, zero-sales checks
│   ├── 03_eda.ipynb                # 16 visual analyses (trends, families, heatmaps)
│   ├── 04_feature_engineering.ipynb# Lag creation, rolling statistics, zero-leakage tests
│   └── 05_model_training.ipynb     # Model training, comparison, evaluation metrics
│
├── src/
│   ├── data_loader.py              # Robust loader, auto-discovery, integrity validator
│   ├── preprocessing.py            # Clean holiday events, interpolate oil, validate zero-sales
│   ├── analytics.py                # Reusable EDA computation and Plotly chart generators
│   ├── feature_engineering.py      # Calendar features, group-wise shifted lags & rolling windows
│   ├── train.py                    # Chronological split, trains Baseline, RF, and XGBoost
│   ├── evaluate.py                 # MAE, RMSE, RMSLE, WAPE, R2, residual analysis
│   ├── predict.py                  # Recursive Daily, Weekly (7d), Monthly (30d) forecasting engine
│   ├── insights.py                 # Deterministic business rule engine on real data
│   └── report_generator.py         # CSV & ReportLab PDF report generation with tables and metrics
│
├── models/
│   ├── best_model.pkl              # Highest-performing model on validation set
│   ├── random_forest.pkl           # Saved Random Forest model
│   ├── xgboost.json                # Saved XGBoost model
│   ├── model_metadata.json         # Training timestamp, hyperparams, feature names, test metrics
│   └── preprocessing_artifacts.pkl # Encoders, scalers, and series history buffers
│
├── dashboard/
│   ├── app.py                      # Main entrypoint, sidebar nav, system status, caching
│   ├── utils.py                    # Shared styles, data caches, model loaders
│   └── pages/
│       ├── 1_Overview.py           # KPIs, sales trends, top performers, executive summary
│       ├── 2_Sales_Analysis.py     # Deep-dive time series trends, promotions, holidays, heatmaps
│       ├── 3_Store_Product_Analysis.py # Geographic, store cluster/type, family rankings
│       ├── 4_Forecasting.py        # Interactive Daily, Weekly, Monthly forecast UI with Plotly
│       ├── 5_Model_Performance.py  # Model comparison table, metrics, residuals, Retrain button
│       └── 6_Reports.py            # Downloadable CSV and PDF reports via ReportLab
│
└── reports/
    ├── data_quality_report.json    # Detailed missingness and deduplication stats
    ├── model_metrics.json          # Metrics across all evaluated models
    └── retail_sales_report.pdf     # Generated PDF report artifact
```

---

## 📦 Dataset Setup & Auto-Discovery

The application automatically checks for the Kaggle Store Sales dataset files in:
1. `data/raw/`
2. `store-sales-time-series-forecasting/`
3. Root workspace directory

Required CSV files:
- `train.csv` (121.8 MB, 3,000,888 records)
- `test.csv` (1.0 MB, 28,512 records)
- `stores.csv` (54 stores across Ecuador)
- `oil.csv` (Daily Brent crude oil benchmark)
- `holidays_events.csv` (National, Regional, and Local public holidays)
- `transactions.csv` (Daily store transaction footfalls)

---

## ⚙️ Installation & Quickstart

### 1. Environment Setup
Python 3.11+ is recommended. Install required libraries:
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data & Engineer Features
Generate the cleaned dataset and leak-free feature matrix:
```bash
python -m src.preprocessing
python -m src.feature_engineering
```

### 3. Train Models
Execute the chronological training pipeline and benchmark all 3 models:
```bash
python -m src.train
```

### 4. Launch the Interactive Dashboard
Run the multi-page Streamlit application:
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Evaluation & Leaderboard

Models are evaluated on an out-of-sample chronological holdout period (**August 1 - August 15, 2017**, 26,730 observations across all 54 stores and 33 product families):

| Model | Split | MAE ($) | RMSE ($) | RMSLE | WAPE | R² Score |
|---|---|---|---|---|---|---|
| **XGBoost Regressor** | **Test** | **$65.53** | **$229.70** | **0.3842** | **0.1409** | **0.9659** |
| Random Forest Regressor | Test | $65.04 | $222.02 | 0.3915 | 0.1398 | 0.9682 |
| Seasonal Naive Baseline | Test | $101.25 | $353.80 | 0.5690 | 0.2177 | 0.9192 |

*XGBoost Regressor was automatically selected as the Champion Model based on superior Validation RMSLE (0.3826).*

---

## 🔮 Forecasting Methodology

The forecasting engine implements **pure recursive multi-step forecasting**:
1. For step $t$, the engine constructs calendar features and retrieves known external covariates (store type, city, holiday status, oil price, promotion assumption).
2. Historical sales lags (`Lag 1`, `Lag 7`, `Lag 14`, `Lag 30`) and rolling statistics (`Rolling Mean 7, 14, 30`, `Rolling Std 7`) are pulled strictly from the historical series buffer.
3. The model predicts $\hat{y}_t = \max(0, \exp(\text{model.predict}(X_t)) - 1)$.
4. The prediction $\hat{y}_t$ is appended to the series sequence.
5. For step $t+1$, `Lag 1` becomes $\hat{y}_t$, `Lag 7` and rolling statistics are recalculated using the updated sequence, and prediction proceeds.
6. This recursion repeats until the full horizon (1 day, 7 days, or 30 days) is completed.

---

## 📈 Strategic Business Insights (Derived from Actual Data)

- **Revenue Concentration**: `GROCERY I` is the enterprise's largest revenue contributor ($343.46M, 32.0% share), followed by `BEVERAGES` ($216.96M, 20.2% share).
- **Flagship Store**: Store #44 in Quito leads all retail outlets with $62.08M in cumulative volume.
- **Promotional Elasticity**: Promoted products generate an average volume of **$1,137.7 vs $158.2** for unpromoted items, reflecting a **7.2x promotional lift factor**.
- **Holiday Shopping Impact**: Public holidays generate an average **+11.9% sales surge** over normal operating weekdays.
- **Intra-Week Seasonality**: **Sunday** represents peak sales footfall with $463 average daily revenue per store-item, followed by Saturday.

---

## 📄 License & Attribution
- Dataset: [Kaggle Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- Built for Academic PBL: Machine Learning & Retail Demand Analytics.
