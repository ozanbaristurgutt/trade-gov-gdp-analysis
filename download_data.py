import io
import pandas as pd
import requests


def main():
    # Hugging Face'teki CSV'nin URL'si
    url = "https://huggingface.co/datasets/tripathyShaswata/GDP-Per-Capita_Gov-Expenditure_Trade/resolve/main/gdp_per_capita.csv"

    print("Downloading CSV from Hugging Face...")

    # Eğer yine SSL hatası alırsan, verify=False kullan (aşağıdaki satırı aç)
    # resp = requests.get(url, timeout=60, verify=False)

    resp = requests.get(url, timeout=60)  # önce normalini dene
    resp.raise_for_status()

    # Gelen text'i pandas'a ver
    csv_text = resp.text
    df = pd.read_csv(io.StringIO(csv_text))

    # Aynı klasöre kaydet
    out_path = "gdp_per_capita.csv"
    df.to_csv(out_path, index=False)

    print("OK, saved to:", out_path)
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
