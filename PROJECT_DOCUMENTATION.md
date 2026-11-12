# Retail Sales Analysis and Prediction Using Machine Learning
## Comprehensive Project Documentation & Technical Architecture Report

---

## 1. Executive Summary

This project delivers an enterprise-grade, end-to-end Machine Learning intelligence and forecasting system built on the **Kaggle Store Sales - Time Series Forecasting** dataset (Corporación Favorita, Ecuador).

The system analyzes over **3,000,888 historical retail transaction records** spanning **54 retail stores** and **33 product families** between **2013 and 2017**, trains and benchmarks multiple machine learning algorithms, implements recursive multi-horizon sales forecasting (Daily, Weekly, Monthly), and surfaces all results through an interactive, multi-page **Streamlit** dashboard with automated **ReportLab** executive PDF report generation.

### Key Performance Highlights
- **Champion Model:** XGBoost Regressor (Histogram split method).
- **Out-of-Sample Test RMSLE:** **0.3842** (Root Mean Squared Logarithmic Error on holdout test set: Aug 1 - Aug 15, 2017).
- **Out-of-Sample Test WAPE:** **0.1409** (Weighted Absolute Percentage Error of ~14.1%).
- **Coefficient of Determination (R2):** **0.9659** (explaining 96.6% of the variance in sales).
- **Promotional Elasticity:** Items on active promotion achieve a **7.2x sales lift** ($1,137.7 vs $158.2).
- **Holiday Impact:** Public holidays generate an average **+11.9% sales surge** over regular weekdays.

---

## 2. Technology Stack & Design Decisions

| Category | Technology | Role & Purpose |
|---|---|---|
| **Programming Language** | Python 3.11+ / 3.13 | Core runtime environment |
| **Data Processing** | Pandas 2.0+, NumPy | Vectorized time-series manipulation, grouped lag operations |
| **Storage Formats** | Apache Parquet (Snappy) | High-speed, columnar, memory-compressed storage (reduced 121MB CSV to 20MB Parquet) |
| **Machine Learning** | Scikit-Learn | Random Forest, OrdinalEncoder, time-series splitting, error metrics |
| **Gradient Boosting** | XGBoost 3.4+ | Fast histogram-based tree boosting with early stopping |
| **Interactive UI** | Streamlit 1.28+ | Multipage dashboard with session caching and real-time inference |
| **Visualizations** | Plotly Express & Graph Objects | Dynamic charts: line trends, scatter, bar, heatmaps, residual plots |
| **Model Persistence** | Joblib & JSON | Serializing trained models, encoders, metadata, and history buffers |
| **PDF Reporting** | ReportLab 5.1+ | Programmatic generation of multi-page executive PDF briefing reports |

---

## 3. Dataset Architecture & Raw Ingestion

The raw dataset comprises 6 interrelated relational tables located in `data/raw/`:

1. **`train.csv` (121.8 MB, 3,000,888 rows):**
   - Columns: `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion`.
   - Granularity: Daily sales per store per product family.
2. **`test.csv` (1.0 MB, 28,512 rows):**
   - Columns: `id`, `date`, `store_nbr`, `family`, `onpromotion`.
   - Period: August 16 to August 31, 2017 (16 days).
3. **`stores.csv` (1.4 KB, 54 rows):**
   - Columns: `store_nbr`, `city`, `state`, `type` (A through E), `cluster` (1 to 17).
4. **`oil.csv` (20.5 KB, 1,218 rows):**
   - Daily Brent crude oil benchmark price (`dcoilwtico`). Significant because Ecuador's economy is highly oil-dependent.
5. **`holidays_events.csv` (22.3 KB, 350 rows):**
   - Columns: `date`, `type`, `locale`, `locale_name`, `description`, `transferred`.
6. **`transactions.csv` (1.5 MB, 83,488 rows):**
   - Daily store-level checkout transaction volume.

### Data Loader Module (`src/data_loader.py`)
- **Auto-Discovery:** Automatically scans candidate directories (`data/raw/`, `store-sales-time-series-forecasting/`, and root) to locate all CSV files.
- **Memory Optimization:** Downcasts types during loading (`int16` for `store_nbr`, `category` for `family`, `float32` for `sales`, `int32` for `onpromotion`), slashing memory footprint by over 60%.

---

## 4. Preprocessing & Data Integrity Module (`src/preprocessing.py`)

A retail time-series dataset presents unique integrity challenges that were systematically resolved:

### A. Continuous Oil Price Imputation
- **Challenge:** The raw `oil.csv` only records market trading days, leaving weekends and national holidays as missing gaps (43 NaNs in trading days, and hundreds of missing calendar dates).
- **Solution:** Reindexed against a continuous 365-day calendar from `2013-01-01` to `2017-08-31`, applied linear interpolation across gaps, and backward/forward-filled boundary days. Result: **0 missing oil values**.

### B. Transferred Holidays & Work Day Logic
- **Challenge:** In Ecuador, when a holiday falls mid-week, it is frequently legally deferred to a Friday or Monday (`transferred == True`). The date where `transferred == True` was an ordinary working day, while the new celebration date appears with `type == 'Transfer'`. Furthermore, government-mandated `Work Day` entries compensate for bridge holidays.
- **Solution:** Excluded `transferred == True` dates and `Work Day` entries from holiday flags.

### C. Overlapping Holiday Deduplication (Cartesian Explosion Prevention)
- **Challenge:** On several dates, multiple holiday events coincide (e.g., a National holiday + Local founding day + Event). A naive join on `date` would duplicate sales records and artificially inflate totals.
- **Solution:** Constructed a hierarchical holiday lookup table sorted by `National > Regional > Local` priority, deduplicated strictly to one primary record per date.
- **Integrity Validation:** Exactly **3,000,888 input rows** merged into exactly **3,000,888 output rows** (`row_count_matched = True`, **0 duplicates**).

### D. Zero-Sales & Negative Sales Preservation
- Zero-sales rows (**939,130 records**, ~31.3%) were identified as legitimate store closures, brand launches, or out-of-stock events and fully preserved.
- Negative sales were investigated: exactly **0 negative values** existed in the historical records.
- Output saved as `data/processed/cleaned_sales.parquet` (20.31 MB) and `reports/data_quality_report.json`.

---

## 5. Leak-Free Feature Engineering (`src/feature_engineering.py`)

In time-series forecasting, data leakage occurs when future or contemporaneous information from the target variable contaminates training inputs. The feature pipeline was architected to guarantee strict mathematical zero leakage:

### Feature Taxonomy

1. **Temporal Calendar Features:**
   - `year`: Integer year (captures macro growth trend).
   - `month` (1-12): Captures annual retail seasonality.
   - `day` (1-31): Monthly payroll cycle effect (salaries paid on 15th and end of month in Ecuador).
   - `dayofweek` (0-6): Weekly shopping patterns.
   - `weekofyear` (1-53): Multi-week trends.
   - `is_weekend` (1/0): Weekend demand surge flag.
   - `quarter` (1-4): Quarterly financial reporting periods.

2. **Group-Wise Shifted Historical Lags:**
   - `sales_lag_1`: Previous day's sales (y_{t-1}).
   - `sales_lag_7`: Sales on the same weekday of the previous week (y_{t-7}).
   - `sales_lag_14`: Sales 2 weeks prior (y_{t-14}).
   - `sales_lag_30`: Sales 1 month prior (y_{t-30}).
   - *Computed independently for each of the 1,782 (store_nbr, family) time series.*

3. **Group-Wise Shifted Rolling Statistics (Zero Target Leakage):**
   - **Crucial Rule:** The sales series is strictly shifted by 1 day (`shift(1)`) *before* calculating rolling windows.
   - `sales_roll_mean_7`: 7-day trailing moving average of past sales.
   - `sales_roll_mean_14`: 14-day trailing moving average.
   - `sales_roll_mean_30`: 30-day trailing moving average.
   - `sales_roll_std_7`: 7-day trailing standard deviation (demand volatility indicator).

4. **External Covariates & Categorical Encoding:**
   - `onpromotion`: Active promotion count.
   - `dcoilwtico`: Brent crude price.
   - `is_holiday`: Binary public holiday indicator.
   - `type`, `cluster`, `city`, `state`, `family`, `holiday_type`, `holiday_locale`: Encoded using `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.

- Output saved as `data/processed/feature_matrix.parquet` (78.05 MB, 30 columns) and `models/preprocessing_artifacts.pkl` (fitted encoder + 60-day historical buffer).

---

## 6. Machine Learning Model Development & Benchmark

### A. Chronological Splitting Strategy
Random train/test splits are invalid for time series because they shuffle future observations into the past. Chronological splitting was implemented:
- **Training Set:** `2016-01-01` to `2017-07-15` (999,702 observations).
- **Validation Set:** `2017-07-16` to `2017-07-31` (28,512 observations, 16 days).
- **Holdout Test Set:** `2017-08-01` to `2017-08-15` (26,730 observations, final 15 days of historical data).

### B. Target Transformation: `log1p(sales)`
Because retail sales exhibit high right-skewness and the Kaggle primary evaluation metric is **RMSLE**, models train on log(1 + y).
Predictions are inverted via exp(y_hat) - 1 and clipped at 0, guaranteeing strictly non-negative sales.

### C. Models Evaluated

1. **Seasonal Naive Baseline (`SeasonalNaiveBaseline`):**
   - Forecasts sales using the same weekday value from 7 days prior (`sales_lag_7`).
   - Benchmark for quantifying whether ML models extract meaningful signal over naive heuristics.
2. **Random Forest Regressor:**
   - 80 trees, `max_depth=16`, `min_samples_leaf=4`, `max_samples=0.7`, `n_jobs=-1`.
   - Ensembled non-linear regression with feature sub-sampling.
3. **XGBoost Regressor (Champion):**
   - 150 boosting rounds, `max_depth=8`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`.
   - `tree_method='hist'` for high-speed multi-core gradient boosting.

### D. Benchmark Results (Final Holdout Test Set: Aug 1 - Aug 15, 2017)

| Model | Split | MAE ($) | RMSE ($) | RMSLE | WAPE | R2 Score |
|---|---|---|---|---|---|---|
| **XGBoost Regressor** | **Test** | **$65.53** | **$229.70** | **0.3842** | **0.1409** | **0.9659** |
| Random Forest Regressor | Test | $65.04 | $222.02 | 0.3915 | 0.1398 | 0.9682 |
| Seasonal Naive Baseline | Test | $101.25 | $353.80 | 0.5690 | 0.2177 | 0.9192 |

**Winner:** XGBoost Regressor outperformed all models with the lowest Validation RMSLE (**0.3826**) and Test RMSLE (**0.3842**), achieving an R2 of **96.6%**.
Persisted to `models/best_model.pkl`, `models/xgboost.json`, and `reports/model_metrics.json`.

---

## 7. Recursive Forecasting Engine (`src/predict.py`)

Forecasting multiple days into the future requires multi-step prediction without access to actual future sales. A **pure recursive multi-step forecasting engine** was implemented:

### Recursive Algorithm:
```
Given: Store S, Product Family F, Horizon H (days), Promotion Assumption P

1. Retrieve historical 60-day sales buffer for (S, F).
2. For day step k in 1 to H:
     a. Compute calendar date features for date t_k.
     b. Retrieve external covariates (Store type, City, Holiday status, Oil price).
     c. Extract sales_lag_1 from the most recent point in the buffer.
     d. Extract sales_lag_7, lag_14, lag_30 from the buffer.
     e. Calculate trailing 7d, 14d, 30d means and 7d standard deviation from the buffer.
     f. Assemble feature vector matching model schema.
     g. Transform categorical features via fitted OrdinalEncoder.
     h. Predict: y_hat_k = max(0, exp(model.predict(X_k)) - 1).
     i. Append (date_k, y_hat_k) to the series buffer.
3. Return complete forecast trajectory dataframe.
```

### Modes Supported:
- **Daily Mode:** Single next-day point forecast.
- **Weekly Mode:** 7-day recursive horizon with weekly total and daily breakdown.
- **Monthly Mode:** 30-day recursive trajectory with weekly breakdown and previous-month comparison.
- **Scenario Testing:** Users can modulate the `promo_assumption` parameter to simulate promotional sales lift.

---

## 8. Strategic Business Insights Module (`src/insights.py`)

A deterministic rule-based insights engine derives real business intelligence from the processed dataset:

1. **Revenue Concentration:** `GROCERY I` generates **$343.46 Million** (32.0% of total revenue), followed by `BEVERAGES` (**$216.96 Million**, 20.2%).
2. **Flagship Store:** **Store #44** located in Quito is the enterprise leader with **$62.08 Million** in cumulative volume.
3. **Promotional Elasticity:** Promoted items average **$1,137.7** in sales vs **$158.2** for non-promoted items (**7.2x promotional lift factor**).
4. **Holiday Lift:** Public holidays produce an average **+11.9% sales increase** over normal weekdays.
5. **Intra-Week Seasonality:** **Sunday** is the peak sales footfall day ($463 avg revenue per item), followed by Saturday.
6. **Store Outlier Alert:** Store #52 recorded sales significantly below regional cluster averages, indicating opportunities for operational restructuring.

---

## 9. Automated PDF & CSV Reporting (`src/report_generator.py`)

- **ReportLab PDF Engine:** Dynamically generates a publication-ready executive briefing PDF (`reports/retail_sales_report.pdf`):
  - Document Title & Timestamp Header.
  - Executive KPI Table (Total Revenue, Average Daily Sales, Active Outlets, Category Count).
  - Machine Learning Model Benchmark Leaderboard Table.
  - Strategic Business Insights Grid.
  - Top 5 Stores and Top 5 Product Categories Table.
- **CSV Data Exporter:** One-click CSV export of forecasts and store/product performance tables.

---

## 10. Multi-Page Interactive Streamlit Dashboard (`dashboard/`)

The application features a modern, executive Streamlit dashboard organized across 6 dedicated pages:

- **`dashboard/app.py` (Home):** Executive welcome, real-time ML engine status, dataset scope, KPI banner, navigation guide, and pipeline architecture diagram.
- **`1_Overview.py` (Page 1):** Enterprise revenue metrics, master daily sales trendline with store/family dropdown filters, top 10 stores & categories, color-coded business insight cards.
- **`2_Sales_Analysis.py` (Page 2):** 4 analytical tabs: Temporal Trendlines (weekly, monthly, yearly), Promotion & Holiday Drivers (elasticity and lift), Seasonality & Intra-week Patterns, Statistical Distributions & Correlation Heatmap.
- **`3_Store_Product_Analysis.py` (Page 3):** Geographic sales analysis (City and State rankings), Store Format comparison (Type A-E share), Cluster analysis, Complete 33-category rankings (Top 10 drivers vs Bottom 10 long-tail opportunities).
- **`4_Forecasting.py` (Page 4):** Interactive forecasting cockpit: Horizon selector (Daily, Weekly, Monthly), Store and Product Family selectors, Promotion assumption slider, Interactive Plotly chart with historical actuals + forecast cone, tabular forecast view, and CSV download.
- **`5_Model_Performance.py` (Page 5):** Champion model banner, interactive leaderboard with split filters, metric cards (RMSLE, WAPE, MAE, RMSE, R2), actual vs predicted curves, residual histogram and scatter diagnostics, feature importance bar chart, and an interactive "Retrain Models Now" button.
- **`6_Reports.py` (Page 6):** Executive report generator: One-click "Generate & Download PDF Report" button (via ReportLab), CSV dataset export selector, and on-screen report data preview tables.

---

## 11. Jupyter Notebooks Suite (`notebooks/`)

Five runnable Jupyter notebooks mirror each stage of the data science lifecycle:
1. `notebooks/01_data_understanding.ipynb`: Exploration of raw CSV schemas, null values, and cardinality.
2. `notebooks/02_preprocessing.ipynb`: Demonstrates oil interpolation, holiday deduplication, and zero-sales preservation.
3. `notebooks/03_eda.ipynb`: Executes 16 visual analyses across time, stores, categories, and promotions.
4. `notebooks/04_feature_engineering.ipynb`: Demonstrates shifted lag creation, rolling windows, and verifies zero target leakage.
5. `notebooks/05_model_training.ipynb`: Chronological model training, evaluation, comparison, and metadata inspection.

---

## 12. Execution Guide & CLI Commands

### 1. Launch Dashboard
```powershell
streamlit run dashboard/app.py
```
Open **http://localhost:8501** in your browser.

### 2. Retrain Models via Terminal
```powershell
python -m src.train
```

### 3. Generate PDF Report via Terminal
```powershell
python -m src.report_generator
```

### 4. Run Preprocessing Pipeline
```powershell
python -m src.preprocessing
python -m src.feature_engineering
```
