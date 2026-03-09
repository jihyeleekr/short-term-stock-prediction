import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TICKERS, MARKET_TICKER


def load_csv(ticker: str) -> pd.DataFrame:
    """
    Load one ticker CSV file
    """
    file_path = RAW_DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(file_path)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names
    """
    df = df.copy()
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLCV columns to numeric
    """
    df = df.copy()

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag/rolling features and targets for one stock
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()

    df["vol_5d"] = df["close"].pct_change().rolling(5).std()
    df["volume_change_1d"] = df["volume"].pct_change(1)

    df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df["target_volatility"] = ((df["close"].shift(-1) - df["close"]) / df["close"]).abs()

    return df


def build_features(spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create market context features from SPY only
    """
    market = spy_df.copy()
    market = market.sort_values("date").reset_index(drop=True)

    market["spy_ret_1d"] = market["close"].pct_change(1)
    market["spy_ret_5d"] = market["close"].pct_change(5)

    return market[["date", "spy_ret_1d", "spy_ret_5d"]]


def prepare_dataframe(ticker: str) -> pd.DataFrame:
    df = load_csv(ticker)
    df = clean_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = convert_numeric_columns(df)
    df["ticker"] = ticker
    return df


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    spy_df = prepare_dataframe(MARKET_TICKER)
    market_features = build_features(spy_df)

    stock_frames = []

    for ticker in TICKERS:
        df = prepare_dataframe(ticker)
        df = add_stock_features(df)
        df = df.merge(market_features, on="date", how="left")
        stock_frames.append(df)

    final_df = pd.concat(stock_frames, ignore_index=True)
    final_df = final_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    final_df = final_df.dropna().reset_index(drop=True)

    output_path = PROCESSED_DATA_DIR / "model_dataset.csv"
    final_df.to_csv(output_path, index=False)

    print(f"Saved processed dataset to: {output_path}")
    print(final_df.head())
    print(final_df.shape)


if __name__ == "__main__":
    main()