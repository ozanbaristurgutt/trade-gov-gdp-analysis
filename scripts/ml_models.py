import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "gdp_per_capita.csv"


def load_and_prepare():
    print("Reading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Aynı temizlik: 1990+ ve ana sütunlarda NA yok
    cols_needed = [
        "GDP per capita",
        "Government expenditure (% of GDP)",
        "Trade as a Share of GDP",
        "Inflation, consumer prices (annual %)",
    ]
    df = df[df["Year"] >= 1990].copy()
    df = df.dropna(subset=cols_needed)

    # Hedef: log(GDP per capita) daha stabil
    df["log_gdp_pc"] = np.log(df["GDP per capita"])

    # Özellikler: trade, gov, inflation, year
    X = df[
        [
            "Trade as a Share of GDP",
            "Government expenditure (% of GDP)",
            "Inflation, consumer prices (annual %)",
            "Year",
        ]
    ].values
    y = df["log_gdp_pc"].values

    print("X shape:", X.shape, "y shape:", y.shape)
    return X, y


def evaluate_model(name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R^2:", r2)

    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}



def main():
    X, y = load_and_prepare()

    # Train / test bölme
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

    results = []

    # 1) Basit Linear Regression (ölçekli)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    y_pred_lin = lin_reg.predict(X_test_scaled)
    results.append(evaluate_model("LinearRegression", y_test, y_pred_lin))

    # 2) Ridge Regression (L2 regularization)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    results.append(evaluate_model("Ridge(alpha=1.0)", y_test, y_pred_ridge))

    # 3) Random Forest (non-linear model)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_model("RandomForestRegressor", y_test, y_pred_rf))

    # Sonuçları tabloya dök
    results_df = pd.DataFrame(results)
    out_path = BASE_DIR / "ml_results.csv"
    results_df.to_csv(out_path, index=False)
    print("\nSaved metrics to:", out_path)

    # Random Forest feature importances (yorum için)
    feature_names = [
        "Trade as a Share of GDP",
        "Government expenditure (% of GDP)",
        "Inflation, consumer prices (annual %)",
        "Year",
    ]
    importances = rf.feature_importances_
    feat_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    feat_path = BASE_DIR / "feature_importances_rf.csv"
    feat_df.to_csv(feat_path, index=False)
    print("Saved feature importances to:", feat_path)
    print("\nRandom Forest feature importances:")
    print(feat_df)


if __name__ == "__main__":
    main()
