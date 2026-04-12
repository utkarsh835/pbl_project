"""
data_loader.py
==============
Downloads and caches historical OHLCV stock data from Yahoo Finance.

Always fetches up to TODAY automatically.
The local CSV cache is refreshed whenever the cached data is more than
CACHE_HOURS old, so the app always shows the latest available prices.
"""

import os
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
TICKERS       = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]
START_DATE    = "2015-01-01"
DATA_DIR      = "data"
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_HOURS   = 6          # re-download if local file is older than this


def _today() -> str:
    """Always returns today's date as an ISO string."""
    return datetime.today().strftime("%Y-%m-%d")


def _cache_path(ticker: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker}.csv")


def _cache_is_stale(path: str) -> bool:
    """True if file doesn't exist or is older than CACHE_HOURS."""
    if not os.path.exists(path):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age > timedelta(hours=CACHE_HOURS)


def download_ticker(ticker: str,
                    start:  str  = START_DATE,
                    end:    str  = None,        # None → today
                    force:  bool = False) -> pd.DataFrame:
    """
    Download OHLCV data for one ticker, caching locally as CSV.

    end=None always resolves to today's date at call time, so
    every fresh download includes the latest trading day.
    """
    if end is None:
        end = _today()

    path = _cache_path(ticker)

    if not force and not _cache_is_stale(path):
        # Cache is fresh — load it, then top-up any missing days
        df_cached = pd.read_csv(path, index_col=0, parse_dates=True)
        last_cached = df_cached.index[-1].strftime("%Y-%m-%d")

        if last_cached < end:
            print(f"[data_loader] Topping up {ticker} from {last_cached} → {end} …")
            new = yf.download(ticker, start=last_cached, end=end,
                              progress=False, auto_adjust=True)
            if isinstance(new.columns, pd.MultiIndex):
                new.columns = new.columns.droplevel(1)
            if not new.empty:
                new = new[REQUIRED_COLS].copy()
                new.index.name = "Date"
                df = pd.concat([df_cached, new])
                df = df[~df.index.duplicated(keep="last")]  # remove overlap
                df.sort_index(inplace=True)
                df.to_csv(path)
                print(f"[data_loader] Updated {ticker} → {df.index[-1].date()}  "
                      f"({len(df)} rows total)")
                return df.dropna()
        else:
            print(f"[data_loader] {ticker} cache is current "
                  f"(latest: {df_cached.index[-1].date()})")
            return df_cached.dropna()

    # Full download
    print(f"[data_loader] Downloading {ticker} {start} → {end} …")
    raw = yf.download(ticker, start=start, end=end,
                      progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    df = raw[REQUIRED_COLS].copy()
    df.index.name = "Date"
    df.to_csv(path)
    print(f"[data_loader] Saved {ticker} → {path}  "
          f"({len(df)} rows, latest: {df.index[-1].date()})")

    df.dropna(inplace=True)
    assert all(c in df.columns for c in REQUIRED_COLS), \
        f"Missing columns in {ticker}: {set(REQUIRED_COLS) - set(df.columns)}"
    return df


def download_all(tickers: list = TICKERS,
                 start:   str  = START_DATE,
                 force:   bool = False) -> dict:
    return {t: download_ticker(t, start=start, force=force) for t in tickers}


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = download_all()
    for ticker, df in datasets.items():
        print(f"\n{ticker}: {df.shape}  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        print(df.tail(3))
