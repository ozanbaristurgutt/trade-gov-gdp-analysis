# Trade, Government Spending, and GDP per Capita Analysis

## 1. Project Overview

This project analyzes how a country’s trade openness and government spending relate to its economic development, measured by GDP per capita. Using a panel dataset of country–year observations, I:

- perform exploratory data analysis (EDA),
- run a small set of hypothesis tests, and
- apply simple machine learning models to predict (log) GDP per capita from macroeconomic features.

Main questions:

- Do countries with higher trade as a share of GDP have higher GDP per capita?
- Is higher government expenditure (% of GDP) associated with higher or lower GDP per capita?
- How does inflation correlate with GDP per capita across countries and over time?
- How well can simple ML models predict GDP per capita using a small set of macro variables?

---

## 2. Data

### 2.1 Primary Dataset (Hugging Face)

- **Source:** Hugging Face Datasets  
- **Slug:** `tripathyShaswata/GDP-Per-Capita_Gov-Expenditure_Trade`  
- **Unit of observation:** Country–year  

Key variables:

- `Entity` (country/region), `Code` (country code), `Year`
- `GDP per capita`
- `Government expenditure (% of GDP)`
- `Trade as a Share of GDP`
- `Inflation, consumer prices (annual %)`

This dataset is suitable because it already combines key macro variables relevant to trade, government spending, inflation, and income for many countries over multiple years.

### 2.2 Additional Variables / Planned Enrichment (Future Work)

I initially planned to merge basic country-level metadata (such as income group and region) from the World Bank’s World Development Indicators (WDI). This would allow comparisons by income group (e.g., low vs. high income) and by region.

In the current submission, I **do not use** these extra variables in the pipeline; they remain a natural extension for future work.

---

## 3. Data Pipeline and Preprocessing

The data are downloaded and processed locally via small Python scripts. Raw data files are **not** stored in the repository; only code, figures, and summary result tables are tracked.

### 3.1 Downloading the Data

- Script: `scripts/download_data.py`

This script:

1. Downloads the CSV file from Hugging Face (`gdp_per_capita.csv`).
2. Saves it locally (e.g., under a `data/` directory).
3. Does not commit the data to Git (data directory is in `.gitignore`).

### 3.2 Cleaning and Analysis-Ready Panel

- Script: `scripts/analysis.py` (also used for EDA + hypothesis tests)

Main preprocessing steps:

- Filter to years **≥ 1990** to focus on a more recent period with better coverage.
- Drop rows with missing values in the four key variables:

  - `GDP per capita`
  - `Government expenditure (% of GDP)`
  - `Trade as a Share of GDP`
  - `Inflation, consumer prices (annual %)`

- Work with the resulting country–year panel for all subsequent analysis.
- For some modeling tasks, define `log_gdp_pc = log(GDP per capita)` to reduce skewness.

The scripts print dataset shapes and variable summaries when run, so the pipeline is transparent and reproducible.

---

## 4. Exploratory Data Analysis (EDA)

EDA is implemented in `scripts/analysis.py` and produces several figures saved under `figures/` (or an equivalent directory).

Key EDA steps:

- Basic summaries (`df.info()`, `df.describe()`) for the cleaned panel.
- Histograms of:

  - GDP per capita
  - Trade as a Share of GDP
  - Government expenditure (% of GDP)
  - Inflation, consumer prices (annual %)

- Time-series plots of GDP per capita for a few selected countries (e.g., Turkey, Germany, United States) to visualize long-run growth patterns.
- A correlation matrix and heatmap for the four main macro variables.

General patterns observed:

- GDP per capita is heavily right-skewed: most country–year observations are at relatively low income levels, with a long tail of high-income observations.
- Trade share, government expenditure, and inflation all show substantial variation across countries and over time, with some extreme inflation episodes.
- The correlation matrix suggests:

  - A positive relationship between trade openness and GDP per capita,
  - A non-trivial association between government expenditure and GDP per capita,
  - A negative relationship between inflation and GDP per capita.

These EDA results motivate the formal hypothesis tests and the subsequent regression/ML models.

---

## 5. Hypothesis Tests

Hypothesis tests are also implemented in `scripts/analysis.py`. They focus on how GDP per capita varies across groups defined by trade openness and inflation, and on simple correlations between government expenditure and GDP.

Main hypotheses:

- **H1 – Trade openness and GDP per capita**

  - Define “low trade” as the bottom 25% of `Trade as a Share of GDP` and “high trade” as the top 25%.
  - Compare mean GDP per capita between these two groups using a Welch two-sample t-test.
  - The script reports group means and p-values; the goal is to see whether high-trade country–year observations systematically have higher income levels.

- **H2 – Government expenditure and GDP per capita**

  - Compute Pearson correlation between `Government expenditure (% of GDP)` and `GDP per capita` (or its log).
  - The script outputs the correlation coefficient and p-value to show the strength and direction of the association.

- **H3 – Inflation and GDP per capita**

  - Split observations into “low inflation” (bottom 25% of `Inflation, consumer prices (annual %)`) and “high inflation” (top 25%).
  - Compare mean GDP per capita between these two groups using a Welch t-test.
  - The goal is to check whether high-inflation environments tend to be associated with lower income levels.

The numerical results (means, t-statistics, p-values) are printed by the script and summarized in the accompanying project report.

---

## 6. Machine Learning Methods and Results

Machine learning experiments are implemented in `scripts/ml_models.py`. The problem is framed as a **regression** task where the target is the logarithm of GDP per capita.

### 6.1 Feature Set and Target

- **Target:**  
  `log(GDP per capita)` for each country–year observation (after filtering and cleaning).

- **Predictors (features):**

  - `Trade as a Share of GDP`
  - `Government expenditure (% of GDP)`
  - `Inflation, consumer prices (annual %)`
  - `Year`

Rows with missing values in these features are removed. The data are split into training and test sets using an 80/20 random split with a fixed seed (`random_state = 42`) for reproducibility.

### 6.2 Models Compared

Three regression models are trained and evaluated:

1. **Linear Regression** (with standardized features)
2. **Ridge Regression** (L2-regularized linear model, alpha = 1.0)
3. **Random Forest Regressor** (200 trees)

### 6.3 Performance Metrics

Performance is evaluated on the held-out test set using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)

The resulting metrics (predicting `log(GDP per capita)`) are:

| Model               | MAE   | RMSE  | R²    |
|---------------------|-------|-------|-------|
| Linear Regression   | 0.694 | 0.874 | 0.462 |
| Ridge (alpha = 1.0) | 0.694 | 0.874 | 0.462 |
| Random Forest       | 0.528 | 0.744 | 0.610 |

(These values are also saved in `results/ml_results.csv`.)

The two linear models perform almost identically, with an R² of about 0.46.  
The Random Forest Regressor clearly improves performance, reducing both MAE and RMSE and increasing R² to about 0.61. This indicates that a non-linear tree-based model captures more of the variation in `log(GDP per capita)` than simple linear models using the same macro predictors.

### 6.4 Feature Importance (Random Forest)

To understand which variables drive the predictions in the best-performing non-linear model, I examine the Random Forest feature importance scores, saved in `results/feature_importances_rf.csv`:

| Feature                               | Importance |
|---------------------------------------|------------|
| Government expenditure (% of GDP)     | 0.535      |
| Trade as a Share of GDP               | 0.199      |
| Inflation, consumer prices (annual %) | 0.179      |
| Year                                  | 0.087      |

These scores suggest that:

- Government expenditure as a share of GDP is the most informative predictor of `log(GDP per capita)` in this simple model.
- Trade openness and inflation also carry substantial information and have similar importance.
- The calendar year variable plays a smaller role, which is expected because cross-country differences and macroeconomic conditions are more influential than a pure time trend in explaining income levels.

---

## 7. Repository Structure

The repository is organized as follows (data directories may be local-only and ignored by git):

```text
.
├── figures/                  # EDA and ML plots (histograms, time series, correlation heatmap, etc.)
├── results/
│   ├── ml_results.csv        # MAE, RMSE, R² for each model
│   └── feature_importances_rf.csv  # Random Forest feature importances
├── scripts/
│   ├── download_data.py      # Download Hugging Face dataset to local disk
│   ├── analysis.py           # EDA + hypothesis tests
│   └── ml_models.py          # ML experiments (regression models)
├── .gitignore                # Excludes local data directory, notebook checkpoints, etc.
└── README.md                 # Project description and summary of results
