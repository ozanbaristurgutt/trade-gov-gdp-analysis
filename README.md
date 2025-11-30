Trade, Government Spending, and GDP per Capita Analysis

## 1. Project Overview

In this project I will analyze how a country’s trade openness and government spending relate to its economic development, measured by GDP per capita. Using a panel dataset of country–year observations, I will first do exploratory data analysis (EDA) and hypothesis testing; later I will build simple machine learning models (mainly regression) to predict GDP per capita from macroeconomic features.

Main questions:
- Do countries with higher trade as a share of GDP have higher GDP per capita?
- Is higher government expenditure (% of GDP) associated with higher or lower GDP per capita?
- How does inflation correlate with GDP per capita across countries and over time?

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

This dataset is suitable because it already combines key macro variables needed for my questions and covers many countries over multiple years.

### 2.2 Planned Enrichment

I plan to merge basic country-level metadata (such as income group and region) from:
- **World Bank / World Development Indicators (WDI)**

These variables will allow comparisons by income group (e.g., low vs high income) and region.

---

## 3. Data Collection Plan

1. **Primary data (Hugging Face):**
   - Use Python (with `datasets` or `pandas`) to download `tripathyShaswata/GDP-Per-Capita_Gov-Expenditure_Trade`.
   - Save the raw file to `data/raw/` (e.g., `gdp_govexp_trade_raw.csv`).
   - Keep a small script (e.g., `scripts/download_hf_data.py`) so the download is reproducible.

2. **Secondary data (World Bank):**
   - Download country income group and region information from the World Bank.
   - Save raw CSVs under `data/raw/`.
   - Write a cleaning/merging script (e.g., `scripts/build_panel.py`) that:
     - Harmonizes country codes,
     - Handles missing values,
     - Produces a final analysis-ready panel in `data/processed/` (e.g., `panel_gdp_trade_govexp.csv`).

---

## 4. Planned Analysis (High-Level)

- **EDA:**  
  - Summary statistics and distributions of GDP per capita, trade share, government expenditure, and inflation.  
  - Time-series plots for selected countries and groups.  
  - Correlation analysis between key variables.

- **Hypothesis testing:**  
  - Compare GDP per capita between high-trade vs low-trade countries (t-tests/ANOVA).  
  - Test association between inflation and GDP per capita (correlation or simple regression).  

- **Machine learning (later stage):**  
  - Regression models (e.g., linear regression and a tree-based model) to predict GDP per capita using trade share, government expenditure, inflation, year, and region/income group.

---

## 5. Planned Repo Structure

- `data/raw/` – Original downloaded data  
- `data/processed/` – Cleaned and merged panel dataset  
- `scripts/` – Downloading and cleaning/merging scripts  
- `notebooks/` – EDA, hypothesis testing, and modeling notebooks  
- `README.md` – Project description and, later, summary of results
