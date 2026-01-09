Şu ana kadar yazdıklarımın hepsi zaten **Markdown**.
`.md` demek “uzantı” demek; içerik normal düz yazı + başlık/tablolu format.
Sorun, senin editörde dosyayı **report.md** olarak kaydetmemen veya içerik yerine yanlış şeyi kopyalaman olabilir.

Aşağıdakinin **tamamını** kopyalayıp `report.md` dosyasına yapıştır, kaydet; bu **doğrudan Markdown dosyası**dır:

---

# Trade, Government Spending, Inflation and GDP per Capita

*Project Report*

---

## 1. Motivation

This project investigates how a few core macroeconomic variables relate to a country’s level of economic development. In particular, it focuses on:

* **Trade openness** (trade as a share of GDP)
* **Government expenditure** (as a share of GDP)
* **Inflation** (consumer prices, annual %)

Textbook macroeconomics usually suggests that:

* More open economies and low inflation are associated with higher income levels,
* The role of government spending is more ambiguous and context-dependent.

The guiding question of this project is:

> How are trade openness, government expenditure (% of GDP), and inflation associated with GDP per capita across countries and over time, and how well can simple machine-learning models predict GDP per capita from these macroeconomic indicators?

The goal is **not** to claim causality. Instead, the aim is to build a transparent pipeline that goes from raw data download to exploratory analysis, hypothesis tests, and basic machine-learning models on a real-world panel dataset.

---

## 2. Data Source

### 2.1 Main dataset

* **Source:** Hugging Face Datasets
* **Dataset slug:** `tripathyShaswata/GDP-Per-Capita_Gov-Expenditure_Trade`
* **Unit of observation:** Country–year

**Variables used in this project**

* `Entity` – country or region name
* `Code` – country code
* `Year`
* `GDP per capita`
* `Government expenditure (% of GDP)`
* `Trade as a Share of GDP`
* `Inflation, consumer prices (annual %)`

The dataset combines these series from World Bank–type sources into a single panel, so I can work with one CSV file instead of manually merging multiple sources.

### 2.2 Data collection process

Raw data are **not** committed to the repository. Instead, I use a small Python script:

* `download_data.py`

This script:

1. Downloads the CSV from the Hugging Face raw URL,
2. Saves it locally as `gdp_per_capita.csv` (in a data directory that is ignored by git),
3. Prints the shape and a short preview for basic sanity checks.

Anyone who clones the repository can reproduce the raw data by running:

```bash
python download_data.py
```

This keeps the repository small while still making the data collection step fully reproducible.

---

## 3. Data Analysis

The analysis has three stages:

1. **Preprocessing and exploratory data analysis (EDA)**
2. **Hypothesis testing**
3. **Machine learning models**

These steps are implemented in Python scripts in the repository (`analysis.py`, `ml_models.py`).

### 3.1 Preprocessing and EDA

**Script:** `analysis.py`

**Preprocessing**

* Filter to **Year ≥ 1990**

  * Focus on the more recent period with better coverage and more comparable macroeconomic structures.
* Drop rows with missing values in the four key variables:

  * GDP per capita
  * Government expenditure (% of GDP)
  * Trade as a Share of GDP
  * Inflation, consumer prices (annual %)

This produces a cleaned country–year panel that is used for both the statistical analysis and the machine-learning part.

**Exploratory data analysis**

The EDA script:

* Prints basic summaries (`info()`, `describe()`)

* Produces histograms for:

  * GDP per capita
  * Trade as a share of GDP
  * Government expenditure (% of GDP)
  * Inflation

* Produces time-series plots of GDP per capita for selected countries (e.g., Turkey, Germany, United States) to visualise long-run income dynamics.

* Computes and plots a correlation matrix for:

  * GDP per capita
  * Trade share
  * Government expenditure as a share of GDP
  * Inflation

The resulting PNG figures (histograms, correlation heatmap, time-series plot) are saved in the repository and provide a general picture of the data before moving to formal tests and models.

---

## 3.2 Hypothesis Testing

Still in `analysis.py`, three simple tests connect visual patterns to more formal statistical statements.

**H1 – Trade openness and GDP per capita**

* Create two groups based on quartiles of `Trade as a Share of GDP`:

  * **Low trade:** bottom 25% of the trade distribution
  * **High trade:** top 25%
* Compare (log) GDP per capita between these two groups using a **Welch two-sample t-test** (allowing for unequal variances).
* The script outputs group means and the p-value.

**H2 – Government expenditure and GDP per capita**

* Compute the **Pearson correlation** between `Government expenditure (% of GDP)` and GDP per capita (or its log).
* This measures whether higher public spending shares are associated with higher or lower income levels on average.

**H3 – Inflation and GDP per capita**

* Split observations by quartiles of `Inflation, consumer prices (annual %)`:

  * **Low inflation:** bottom 25%
  * **High inflation:** top 25%
* Use another Welch t-test to compare GDP per capita between these two groups.
* The question is whether high-inflation environments tend to correspond to lower income levels.

The focus is on the **direction** and **statistical strength** of the relationships, rather than detailed causal identification.

---

## 3.3 Machine Learning Methods

**Script:** `ml_models.py`

Here the problem is treated as a **regression task**.

**Target variable**

* `log_gdp_pc = log(GDP per capita)`

  * Taking logs reduces skewness and makes errors roughly comparable across low- and high-income observations.

**Feature set (X)**

* `Trade as a Share of GDP`
* `Government expenditure (% of GDP)`
* `Inflation, consumer prices (annual %)`
* `Year`

Observations with missing values in these features are dropped.

**Train–test split**

* 80% **training**, 20% **test**
* Random split with `random_state = 42` to ensure reproducibility.

**Models evaluated**

Three standard regression models are compared:

1. **Linear Regression** (with standardized features)
2. **Ridge Regression** (L2-regularized linear model, `alpha = 1.0`)
3. **Random Forest Regressor** (200 trees, default hyperparameters otherwise)

For Linear and Ridge, the features are standardized using `StandardScaler`. The Random Forest uses the original scale.

**Evaluation metrics**

On the held-out test set, the following metrics are computed:

* Mean Absolute Error (**MAE**)
* Root Mean Squared Error (**RMSE**)
* Coefficient of determination (**R²**)

These metrics are saved into `ml_results.csv`. Random Forest feature importances are saved into `feature_importances_rf.csv`.

---

## 4. Findings

### 4.1 Descriptive and Statistical Findings

From EDA plus the three hypothesis tests, several patterns emerge:

* **Income distribution is highly unequal.**
  GDP per capita is heavily right-skewed: most country–year observations are concentrated at relatively low income levels, with a long tail of high-income countries.

* **Trade openness is positively associated with income.**
  The high-trade group (top 25% of trade share) has substantially higher average GDP per capita than the low-trade group (bottom 25%). The Welch t-test rejects equality of means, supporting the view that more open economies tend to be richer (even though causality is not established).

* **High inflation is associated with lower income.**
  The high-inflation group has significantly lower GDP per capita than the low-inflation group. This is consistent with the idea that persistent high inflation is a sign of macroeconomic instability and goes together with lower development.

* **Government expenditure is positively, but moderately, related to income.**
  The Pearson correlation between `Government expenditure (% of GDP)` and GDP per capita is positive but not extremely large. Richer countries often have larger public sectors as a share of GDP, but the relationship is clearly not one-to-one and likely depends on institutional and demographic factors.

In short, the statistical analysis confirms that this dataset contains patterns that broadly match standard macroeconomic narratives: high trade and low inflation are associated with higher GDP per capita, while the role of government spending is more nuanced but tends to be positively correlated with income.

---

### 4.2 Machine Learning Results

On the test set (predicting `log(GDP per capita)`), the models achieve the following performance:

| Model               | MAE   | RMSE  | R²    |
| ------------------- | ----- | ----- | ----- |
| Linear Regression   | 0.694 | 0.874 | 0.462 |
| Ridge (alpha = 1.0) | 0.694 | 0.874 | 0.462 |
| Random Forest       | 0.528 | 0.744 | 0.610 |

**Interpretation**

* **Linear vs Ridge**
  Linear Regression and Ridge Regression perform almost identically. Given this feature set and sample size, the plain linear model does not appear to overfit heavily; the L2 penalty in Ridge does not noticeably improve performance.

* **Random Forest**
  The Random Forest Regressor clearly improves over the linear models:

  * MAE improves from ~0.69 to ~0.53,
  * RMSE improves from ~0.87 to ~0.74,
  * R² increases from ~0.46 to ~0.61.

  This indicates that allowing non-linearities and interactions (through trees) captures additional structure in the relationship between trade, government expenditure, inflation, time, and income.

---

### 4.3 Feature Importance

Random Forest feature importance scores (normalized to sum to 1):

| Feature                               | Importance |
| ------------------------------------- | ---------- |
| Government expenditure (% of GDP)     | 0.535      |
| Trade as a Share of GDP               | 0.199      |
| Inflation, consumer prices (annual %) | 0.179      |
| Year                                  | 0.087      |

**Interpretation**

* **Government expenditure** is the most informative predictor in this model. Within this limited feature set, cross-country differences in the size of the public sector explain a large share of the variation in `log(GDP per capita)`.
* **Trade openness** and **inflation** also carry important information and have similar importance values. They provide additional structure beyond government expenditure alone.
* **Year** has the lowest importance: once we condition on the macroeconomic variables themselves, a pure time trend adds relatively little predictive power.

Overall, even with only four predictors, a non-linear model can explain a meaningful fraction of the variation in income levels across countries and years.

---

## 5. Limitations and Future Work

This project is intentionally simple and has several important limitations.

1. **Limited feature set**
   Only four predictors are used (trade share, government expenditure, inflation, year). Many other important determinants of income are missing—education, human capital, institutions, geography, demographics, natural resources, etc.—so the models are far from fully explanatory.

2. **No explicit panel structure in ML**
   The regression models treat all rows as independent observations. They do not include country fixed effects or random effects, so they cannot disentangle within-country dynamics from cross-country differences. A large part of the signal is driven by persistent gaps between countries.

3. **Correlational, not causal**
   All relationships are observational. High trade and low inflation are associated with higher income, but this does not mean that changing trade policy or inflation will directly cause income to change. Reverse causality and omitted variables are both plausible.

4. **Data quality and measurement**
   The underlying data rely on international organizations. Measurement error (especially for inflation and government expenditure in lower-income countries) and missing data may affect the results.

5. **Simple models and evaluation protocol**
   Standard baseline models are used (linear regression, Ridge, Random Forest) and a single random train–test split. More robust evaluation (e.g., repeated cross-validation, country-level cross-validation, hyperparameter tuning) could change the reported metrics.

---

### Future work

Possible extensions of this project include:

* **Richer feature sets**
  Adding variables such as investment rates, human capital indicators, governance indices, or demographic measures would allow more realistic models of income determination.

* **Panel or hierarchical models**
  Using country fixed-effects regression, random-effects models, or hierarchical models could better separate within-country changes over time from cross-country differences.

* **Classification framing**
  Discretizing income into categories (e.g., low / middle / high income) would allow classification models and the use of evaluation tools like precision–recall curves and confusion matrices.

* **Towards causal inference**
  With stronger assumptions or suitable instruments, one could attempt quasi-experimental designs (e.g., using policy shocks or external instruments for trade or inflation) to move from correlation toward causal claims.

Despite these limitations, the project demonstrates an end-to-end pipeline—from raw data download, through EDA and hypothesis tests, to basic ML modeling—using a publicly available macroeconomic dataset and transparent Python code that can be easily rerun from the repository.

