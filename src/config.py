from pathlib import Path

# Base path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

# Stock universe for initial MVP
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
MARKET_TICKER= "SPY"

# Date range
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"