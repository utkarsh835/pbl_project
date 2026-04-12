"""
preprocessing.py
================
Predict DAILY RETURN (% change) instead of absolute price.
This eliminates scaler drift — the model never needs to guess
what $260 vs $180 means; it just learns +1.2% or -0.8%.

Price reconstruction:  predicted_price = last_close × (1 + predicted_return)
Direction:             sign of predicted_return  (always consistent with price)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from pattern_detection import detect_all_patterns, PATTERN_NAMES

WINDOW_SIZE  = 30
OHLCV_COLS   = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLS = None   # set dynamically after engineering
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build input features. Everything is expressed as RETURNS / ratios
    so the model is scale-invariant across time and tickers.
    """
    feat = pd.DataFrame(index=df.index)

    # Price-based returns (scale-free)
    feat["ret_1d"]   = df["Close"].pct_change()
    feat["ret_3d"]   = df["Close"].pct_change(3)
    feat["ret_5d"]   = df["Close"].pct_change(5)
    feat["ret_10d"]  = df["Close"].pct_change(10)

    # Intraday structure (ratios, not absolutes)
    feat["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    feat["body"]     = (df["Close"] - df["Open"]).abs() / df["Close"]
    feat["gap"]      = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Volume change
    feat["vol_chg"]  = df["Volume"].pct_change()

    # Volatility (rolling std of returns)
    feat["vol_5d"]   = feat["ret_1d"].rolling(5).std()
    feat["vol_10d"]  = feat["ret_1d"].rolling(10).std()

    # Candlestick patterns (already 0/1, scale-free)
    patterns = detect_all_patterns(df)
    feat = pd.concat([feat, patterns], axis=1)

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    return feat


# Column groups
def _cont_cols(feat):
    return [c for c in feat.columns if c not in PATTERN_NAMES]

def _pat_cols():
    return PATTERN_NAMES


class DataScaler:
    """RobustScaler on continuous features only. Patterns stay as 0/1."""

    def __init__(self):
        self.scaler = RobustScaler()
        self._cont  = None

    def fit_transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        self._cont = _cont_cols(feat)
        result = feat.copy()
        result[self._cont] = self.scaler.fit_transform(feat[self._cont])
        return result

    def transform(self, feat: pd.DataFrame) -> pd.DataFrame:
        result = feat.copy()
        result[self._cont] = self.scaler.transform(feat[self._cont])
        return result


def make_windows(feat_scaled: pd.DataFrame,
                 df_raw: pd.DataFrame,
                 window: int = WINDOW_SIZE):
    """
    Build sliding windows. Target is NEXT-DAY RETURN (%, unscaled).

    Returns
    -------
    X          : ndarray (N, window, n_features)  — all features combined
    y_return   : ndarray (N,)   — next-day % return (raw, not scaled)
    y_direction: ndarray (N,)   — 1=Up, 0=Down
    last_closes: ndarray (N,)   — the raw Close on prediction day (for price reconstruction)
    dates      : list
    """
    all_cols = list(feat_scaled.columns)
    arr      = feat_scaled.values

    # Align raw close with the scaled feature index
    close_aligned = df_raw["Close"].reindex(feat_scaled.index)

    X_list, y_ret_list, y_dir_list, last_close_list, dates = [], [], [], [], []

    n = len(feat_scaled)
    for i in range(n - window):
        X_list.append(arr[i : i + window])

        # next-day return in actual % (e.g. 0.012 = +1.2%)
        c_now  = close_aligned.iloc[i + window - 1]
        c_next = close_aligned.iloc[i + window]
        ret    = (c_next - c_now) / c_now

        y_ret_list.append(ret)
        y_dir_list.append(1 if ret > 0 else 0)
        last_close_list.append(c_now)
        dates.append(feat_scaled.index[i + window])

    return (np.array(X_list,          dtype=np.float32),
            np.array(y_ret_list,       dtype=np.float32),
            np.array(y_dir_list,       dtype=np.float32),
            np.array(last_close_list,  dtype=np.float32),
            dates)


def split_dataset(X, y_ret, y_dir, last_closes, dates,
                  train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    n       = len(y_ret)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    splits = {}
    for tag, sl in [("train", slice(0, i_train)),
                    ("val",   slice(i_train, i_val)),
                    ("test",  slice(i_val, None))]:
        splits[tag] = {
            "X":           X[sl],
            "y_return":    y_ret[sl],
            "y_direction": y_dir[sl],
            "last_closes": last_closes[sl],
            "dates":       dates[sl.start : sl.stop],
        }
        print(f"  {tag:5s}: {len(splits[tag]['y_return']):5d} samples  "
              f"{splits[tag]['dates'][0]} → {splits[tag]['dates'][-1]}")
    return splits


def prepare_data(df: pd.DataFrame, window: int = WINDOW_SIZE):
    print("[preprocessing] Building feature frame …")
    feat = build_feature_frame(df)
    print(f"  Features: {list(feat.columns)}")

    scaler      = DataScaler()
    feat_scaled = scaler.fit_transform(feat)

    print(f"[preprocessing] Creating windows (window={window}) …")
    X, y_ret, y_dir, last_closes, dates = make_windows(feat_scaled, df, window)
    print(f"  Total windows: {len(y_ret)}")

    splits = split_dataset(X, y_ret, y_dir, last_closes, dates)
    return splits, scaler, feat_scaled.shape[1]   # also return n_features


if __name__ == "__main__":
    from data_loader import download_ticker
    df = download_ticker("AAPL")
    splits, scaler, n_feat = prepare_data(df)
    tr = splits["train"]
    print(f"\nX shape      : {tr['X'].shape}")
    print(f"y_return (5) : {tr['y_return'][:5]}")
    print(f"last_closes  : {tr['last_closes'][:5]}")