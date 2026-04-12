"""
pattern_detection.py
====================
Rule-based candlestick pattern detection using OHLC data.

Each detector returns a binary Series (1 = pattern present, 0 = absent).

Patterns implemented
--------------------
  1. Doji
  2. Hammer
  3. Shooting Star
  4. Bullish Engulfing
  5. Bearish Engulfing
  6. Morning Star
  7. Evening Star

Reference thresholds are based on widely-used technical-analysis conventions.
Every function is self-contained so it can be tested or extended independently.
"""

import numpy as np
import pandas as pd


# ── Column-flattening fix ─────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns like ('Close', 'AAPL').
    This flattens them to plain strings ('Close') so every helper works
    regardless of how the DataFrame was created.
    Also squeezes any single-column DataFrames back to a plain Series.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)   # keep first level only
    return df


# ── Low-level OHLC helpers ────────────────────────────────────────────────────

def _body(df: pd.DataFrame) -> pd.Series:
    """Absolute size of the candle body."""
    return (df["Close"] - df["Open"]).abs().squeeze()


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    """Upper wick length."""
    return (df["High"] - df[["Open", "Close"]].max(axis=1)).squeeze()


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    """Lower wick length."""
    return (df[["Open", "Close"]].min(axis=1) - df["Low"]).squeeze()


def _range(df: pd.DataFrame) -> pd.Series:
    """Total candle range (High - Low)."""
    return (df["High"] - df["Low"]).squeeze()


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    """True where Close > Open (green candle)."""
    return (df["Close"] > df["Open"]).squeeze()


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    """True where Close < Open (red candle)."""
    return (df["Close"] < df["Open"]).squeeze()


# ── Individual pattern detectors ──────────────────────────────────────────────

def detect_doji(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """
    Doji  –  body is very small relative to the full candle range.
    Signal: indecision / potential reversal.

    Parameters
    ----------
    threshold : body / range ratio below which the day is a Doji (default 5 %).
    """
    df    = _flatten(df)
    rng   = _range(df).replace(0, np.nan)
    ratio = _body(df) / rng
    return pd.Series((ratio < threshold).astype(int).values,
                     index=df.index, name="Doji")


def detect_hammer(df: pd.DataFrame,
                  body_ratio:   float = 0.3,
                  shadow_ratio: float = 2.0) -> pd.Series:
    df     = _flatten(df)
    body   = _body(df)
    rng    = _range(df).replace(0, np.nan)
    lower  = _lower_shadow(df)
    upper  = _upper_shadow(df)

    cond = (
        (body / rng <= body_ratio) &
        (lower >= shadow_ratio * body.replace(0, np.nan)) &
        (upper <= body)
    )
    return pd.Series(cond.fillna(False).astype(int).values,
                     index=df.index, name="Hammer")


def detect_shooting_star(df: pd.DataFrame,
                         body_ratio:   float = 0.3,
                         shadow_ratio: float = 2.0) -> pd.Series:
    df    = _flatten(df)
    body  = _body(df)
    rng   = _range(df).replace(0, np.nan)
    lower = _lower_shadow(df)
    upper = _upper_shadow(df)

    cond = (
        (body / rng <= body_ratio) &
        (upper >= shadow_ratio * body.replace(0, np.nan)) &
        (lower <= body)
    )
    return pd.Series(cond.fillna(False).astype(int).values,
                     index=df.index, name="ShootingStar")


def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    df           = _flatten(df)
    prev_bearish = _is_bearish(df).shift(1)
    curr_bullish = _is_bullish(df)
    open_below   = df["Open"].squeeze()  < df["Close"].squeeze().shift(1)
    close_above  = df["Close"].squeeze() > df["Open"].squeeze().shift(1)

    cond = prev_bearish & curr_bullish & open_below & close_above
    return pd.Series(cond.fillna(False).astype(int).values,
                     index=df.index, name="BullishEngulfing")


def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    df           = _flatten(df)
    prev_bullish = _is_bullish(df).shift(1)
    curr_bearish = _is_bearish(df)
    open_above   = df["Open"].squeeze()  > df["Close"].squeeze().shift(1)
    close_below  = df["Close"].squeeze() < df["Open"].squeeze().shift(1)

    cond = prev_bullish & curr_bearish & open_above & close_below
    return pd.Series(cond.fillna(False).astype(int).values,
                     index=df.index, name="BearishEngulfing")


def detect_morning_star(df: pd.DataFrame,
                        body_ratio: float = 0.3) -> pd.Series:
    df   = _flatten(df)
    rng  = _range(df).replace(0, np.nan)
    body = _body(df)

    large_bearish_d2 = _is_bearish(df).shift(2) & (body.shift(2) / rng.shift(2) > body_ratio)
    small_body_d1    = (body.shift(1) / rng.shift(1)) < body_ratio
    bullish_d0       = _is_bullish(df)

    mid_d2          = (df["Open"].squeeze().shift(2) + df["Close"].squeeze().shift(2)) / 2
    close_above_mid = df["Close"].squeeze() > mid_d2

    cond = large_bearish_d2 & small_body_d1 & bullish_d0 & close_above_mid
    return pd.Series(cond.fillna(False).astype(int).values,
                     index=df.index, name="MorningStar")


def detect_evening_star(df: pd.DataFrame,
                        body_ratio: float = 0.3) -> pd.Series:
    df   = _flatten(df)
    rng  = _range(df).replace(0, np.nan)
    body = _body(df)

    large_bullish_d2 = _is_bullish(df).shift(2) & (body.shift(2) / rng.shift(2) > body_ratio)
    small_body_d1    = (body.shift(1) / rng.shift(1)) < body_ratio
    bearish_d0       = _is_bearish(df)

    mid_d2           = (df["Open"].squeeze().shift(2) + df["Close"].squeeze().shift(2)) / 2
    close_below_mid  = df["Close"].squeeze() < mid_d2

    cond = large_bullish_d2 & small_body_d1 & bearish_d0 & close_below_mid
    return pd.Series(cond.fillna(False).astype(int).values,
                     index=df.index, name="EveningStar")


# ── Main entry point ──────────────────────────────────────────────────────────

PATTERN_NAMES = [
    "Doji",
    "Hammer",
    "ShootingStar",
    "BullishEngulfing",
    "BearishEngulfing",
    "MorningStar",
    "EveningStar",
]


def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every detector and return a DataFrame of binary flags.

    Parameters
    ----------
    df : DataFrame with at least columns [Open, High, Low, Close]

    Returns
    -------
    pd.DataFrame  – same index as `df`, columns = PATTERN_NAMES, values in {0, 1}
    """
    df = _flatten(df)   # normalise MultiIndex columns from yfinance
    patterns = pd.concat([
        detect_doji(df),
        detect_hammer(df),
        detect_shooting_star(df),
        detect_bullish_engulfing(df),
        detect_bearish_engulfing(df),
        detect_morning_star(df),
        detect_evening_star(df),
    ], axis=1)

    return patterns


def get_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility: count how often each pattern fires and its hit-rate.
    Useful for exploratory analysis.
    """
    patterns = detect_all_patterns(df)
    n = len(patterns)
    summary = pd.DataFrame({
        "count":    patterns.sum(),
        "hit_rate": (patterns.sum() / n * 100).round(2),
    })
    return summary


def describe_latest_patterns(df: pd.DataFrame) -> dict:
    """
    Return a dict describing which patterns fired on the most recent day.
    Used by the Streamlit app for the 'highlight' feature.
    """
    patterns = detect_all_patterns(df)
    latest   = patterns.iloc[-1]
    fired    = [col for col in PATTERN_NAMES if latest[col] == 1]
    return {
        "date":    df.index[-1].strftime("%Y-%m-%d"),
        "fired":   fired if fired else ["None detected"],
        "all":     latest.to_dict(),
    }


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    raw = yf.download("AAPL", start="2023-01-01", end="2024-01-01",
                      progress=False, auto_adjust=True)

    print("Pattern frequency for AAPL (2023):")
    print(get_pattern_summary(raw))

    print("\nLatest day patterns:")
    print(describe_latest_patterns(raw))