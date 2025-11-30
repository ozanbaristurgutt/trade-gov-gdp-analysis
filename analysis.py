import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Bu dosyanın bulunduğu klasör
BASE_DIR = Path(__file__).resolve().parent
print("BASE_DIR:", BASE_DIR)

plt.rcParams["figure.figsize"] = (10, 6)


def load_and_clean(filename="gdp_per_capita.csv") -> pd.DataFrame:
    csv_path = BASE_DIR / filename
    print("Reading CSV from:", csv_path)

    df = pd.read_csv(csv_path)

    print("\n--- RAW INFO ---")
    print(df.info())

    cols_needed = [
        "GDP per capita",
        "Government expenditure (% of GDP)",
        "Trade as a Share of GDP",
        "Inflation, consumer prices (annual %)",
    ]

    df_sub = df[df["Year"] >= 1990].copy()
    df_sub = df_sub.dropna(subset=cols_needed)

    print("\n--- AFTER FILTER & DROPNA ---")
    print(df_sub.info())
    print(df_sub.describe().T)

    return df_sub


def do_eda(df_sub: pd.DataFrame):
    print("\n=== EDA: histograms ===")

    def hist(col, title, fname):
        sns.histplot(df_sub[col], bins=50)
        plt.title(title)
        plt.tight_layout()
        out_path = BASE_DIR / fname
        plt.savefig(out_path)
        plt.close()
        print("Saved:", out_path)

    hist("GDP per capita", "GDP per capita (1990+)", "hist_gdp_per_capita.png")
    hist(
        "Trade as a Share of GDP",
        "Trade as a Share of GDP (1990+)",
        "hist_trade_share.png",
    )
    hist(
        "Government expenditure (% of GDP)",
        "Government expenditure (% of GDP) (1990+)",
        "hist_gov_exp.png",
    )
    hist(
        "Inflation, consumer prices (annual %)",
        "Inflation (annual %) (1990+)",
        "hist_inflation.png",
    )

    print("\n=== EDA: time series (Turkey, Germany, United States) ===")
    countries = ["Turkey", "Germany", "United States"]
    plt.clf()
    for c in countries:
        sub = df_sub[df_sub["Entity"] == c]
        if sub.empty:
            print("No data for:", c)
            continue
        plt.plot(sub["Year"], sub["GDP per capita"], marker="o", label=c)

    plt.legend()
    plt.title("GDP per capita over time (1990+)")
    plt.xlabel("Year")
    plt.ylabel("GDP per capita")
    plt.tight_layout()
    out_ts = BASE_DIR / "timeseries_gdp_selected_countries.png"
    plt.savefig(out_ts)
    plt.close()
    print("Saved:", out_ts)

    cols = [
        "GDP per capita",
        "Government expenditure (% of GDP)",
        "Trade as a Share of GDP",
        "Inflation, consumer prices (annual %)",
    ]
    corr = df_sub[cols].corr()
    print("\n=== Correlation matrix ===")
    print(corr)

    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation matrix")
    plt.tight_layout()
    out_corr = BASE_DIR / "corr_matrix.png"
    plt.savefig(out_corr)
    plt.close()
    print("Saved:", out_corr)


def hypothesis_tests(df_sub: pd.DataFrame):
    print("\n=== Hypothesis tests ===")

    # H1: High trade vs low trade -> GDP per capita
    trade = df_sub["Trade as a Share of GDP"]
    q25 = trade.quantile(0.25)
    q75 = trade.quantile(0.75)

    low_trade = df_sub[trade <= q25]["GDP per capita"]
    high_trade = df_sub[trade >= q75]["GDP per capita"]

    print("\nH1: High trade vs low trade (GDP per capita)")
    print("low_trade mean:", low_trade.mean())
    print("high_trade mean:", high_trade.mean())
    t_stat, p_val = stats.ttest_ind(high_trade, low_trade, equal_var=False)
    print("t-stat:", t_stat, "p-val:", p_val)

    # H2: Gov. exp vs GDP correlation
    print("\nH2: Government expenditure vs GDP per capita (Pearson corr)")
    gov = df_sub["Government expenditure (% of GDP)"]
    gdp = df_sub["GDP per capita"]
    mask = gov.notna() & gdp.notna()
    corr_gov, p_gov = stats.pearsonr(gov[mask], gdp[mask])
    print("corr:", corr_gov, "p-val:", p_gov)

    # H3: High vs low inflation -> GDP per capita
    print("\nH3: High inflation vs low inflation (GDP per capita)")
    infl = df_sub["Inflation, consumer prices (annual %)"]
    q25_infl = infl.quantile(0.25)
    q75_infl = infl.quantile(0.75)

    low_infl = df_sub[infl <= q25_infl]["GDP per capita"]
    high_infl = df_sub[infl >= q75_infl]["GDP per capita"]

    print("low_infl mean:", low_infl.mean())
    print("high_infl mean:", high_infl.mean())
    t_stat_infl, p_val_infl = stats.ttest_ind(
        low_infl, high_infl, equal_var=False
    )
    print("t-stat:", t_stat_infl, "p-val:", p_val_infl)


def main():
    df_sub = load_and_clean("gdp_per_capita.csv")
    do_eda(df_sub)
    hypothesis_tests(df_sub)


if __name__ == "__main__":
    main()
