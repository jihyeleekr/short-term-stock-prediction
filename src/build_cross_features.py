import pandas as pd
from config import PROCESSED_DATA_DIR


TARGET_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
FEATURE_STOCKS = ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM"]


def build_cross_dataset(df):
    """
    Create dataset where each row contains all stock returns as features
    """

    # pivot: rows = date, columns = ticker
    pivot = df.pivot(index="date", columns="ticker", values="ret_1d")

    pivot = pivot.reset_index()

    return pivot


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    pivot_df = build_cross_dataset(df)

    print(pivot_df.head())

    # save
    pivot_df.to_csv(PROCESSED_DATA_DIR / "cross_asset_dataset.csv", index=False)


if __name__ == "__main__":
    main()