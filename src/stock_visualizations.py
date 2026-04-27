import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR, OUTPUT_DIR


TARGET_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def get_stock_figure_dir(ticker: str):
    """
    Create and return a separate figure directory for each stock.
    Example: outputs/figures/AAPL/
    """
    stock_dir = FIGURE_DIR / ticker
    stock_dir.mkdir(parents=True, exist_ok=True)
    return stock_dir


def plot_price(df: pd.DataFrame, ticker: str):
    """
    Plot and save closing price over time.
    """
    df_ticker = df[df["ticker"] == ticker].copy()
    stock_dir = get_stock_figure_dir(ticker)

    plt.figure(figsize=(9, 4))

    sns.lineplot(
        data=df_ticker,
        x="date",
        y="close",
    )

    plt.title(f"{ticker} Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = stock_dir / f"{ticker}_price.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_returns(df: pd.DataFrame, ticker: str):
    """
    Plot and save daily returns over time.
    """
    df_ticker = df[df["ticker"] == ticker].copy()
    stock_dir = get_stock_figure_dir(ticker)

    plt.figure(figsize=(9, 4))

    sns.lineplot(
        data=df_ticker,
        x="date",
        y="ret_1d",
    )

    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = stock_dir / f"{ticker}_returns.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def plot_volatility(df: pd.DataFrame, ticker: str):
    """
    Plot and save 5-day rolling volatility.
    """
    df_ticker = df[df["ticker"] == ticker].copy()
    stock_dir = get_stock_figure_dir(ticker)

    plt.figure(figsize=(9, 4))

    sns.lineplot(
        data=df_ticker,
        x="date",
        y="vol_5d",
    )

    plt.title(f"{ticker} 5-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("5-Day Rolling Volatility")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = stock_dir / f"{ticker}_volatility.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    dataset_path = PROCESSED_DATA_DIR / "model_dataset.csv"

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    for ticker in TARGET_STOCKS:
        print("\n" + "=" * 60)
        print(f"Creating stock visualizations for {ticker}")
        print("=" * 60)

        plot_price(df, ticker)
        plot_returns(df, ticker)
        plot_volatility(df, ticker)


if __name__ == "__main__":
    main()