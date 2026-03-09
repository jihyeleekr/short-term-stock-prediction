import pandas as pd
import yfinance as yf

from config import RAW_DATA_DIR, TICKERS, MARKET_TICKER, START_DATE, END_DATE

def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
  """
  Download daily OHLCV data from one ticker using Yahoo Finace API.
  """

  df = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    progress=False
  )

  if df.empty:
    raise ValueError(f"No data returned for ticker: {ticker}")
  
  df = df.rest_index()
  df["Ticker"] = ticker
  return df

def save_data(df: pd.DataFrame, ticker: str) -> None:
  """ 
  Save a ticker dataframe to CSV file
  """

  RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
  file_path = RAW_DATA_DIR / f"{ticker}.csv"
  df.to_csv(file_path, index=False)
  print(f"Saved: {file_path}")

def main() -> None:
  all_tickers = TICKERS + [MARKET_TICKER]

  for ticker in all_tickers:
    print(f"Downloading {ticker}...")
    df = download_data(ticker, START_DATE, END_DATE)
    save_data(df, ticker)

  print("Data collection complete.")

if __name__ == "__main__":
  main()