"""
compare_models.py
=================
Runs classical + ML baseline models on the same test set as the Hybrid CNN+LSTM
and produces a side-by-side comparison table + bar charts.

Run AFTER train.py has already saved the scaler and splits.

Usage
-----
    python compare_models.py --ticker AAPL --window 30
"""

import os
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA

from data_loader    import download_ticker
from preprocessing  import prepare_data, WINDOW_SIZE


# ── Helpers ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def direction_accuracy(y_true_price, y_pred_price):
    """
    Accuracy of predicting whether price goes UP or DOWN vs previous day.
    Uses consecutive differences on the prediction array itself.
    """
    true_dir = (np.diff(y_true_price) > 0).astype(int)
    pred_dir = (np.diff(y_pred_price) > 0).astype(int)
    return float(accuracy_score(true_dir, pred_dir))


# ── Flatten windows to 2-D for sklearn models ─────────────────────────────────

def flatten(X_ohlcv, X_patterns):
    """Concatenate + flatten (samples, window, features) → (samples, window*features)."""
    n = X_ohlcv.shape[0]
    return np.concatenate([
        X_ohlcv.reshape(n, -1),
        X_patterns.reshape(n, -1)
    ], axis=1)


# ── Model 1: Naïve baseline ───────────────────────────────────────────────────

def naive_baseline(scaler, splits):
    """
    Predict tomorrow's price = today's price (no model at all).
    The simplest possible benchmark — any real model must beat this.
    """
    te        = splits["test"]
    true_sc   = te["price"]
    true_real = scaler.inverse_close(true_sc)

    # Shift by 1: predict[i] = true[i-1]
    pred_real = np.roll(true_real, 1)
    pred_real[0] = true_real[0]   # first prediction = first actual

    return true_real, pred_real


# ── Model 2: ARIMA ───────────────────────────────────────────────────────────

def arima_baseline(df_close_train, df_close_test):
    """
    Fit ARIMA(5,1,0) on the training close series, then forecast
    one step at a time over the test period (walk-forward).
    """
    print("  [ARIMA] Fitting walk-forward ARIMA(5,1,0) … (this takes ~30 sec)")
    history = list(df_close_train)
    preds   = []

    for i, actual in enumerate(df_close_test):
        try:
            model = ARIMA(history, order=(5, 1, 0))
            fit   = model.fit()
            yhat  = fit.forecast(steps=1)[0]
        except Exception:
            yhat  = history[-1]   # fallback: naïve
        preds.append(yhat)
        history.append(actual)   # walk-forward: add true value

        if (i + 1) % 50 == 0:
            print(f"    … {i+1}/{len(df_close_test)} steps done")

    return np.array(preds)


# ── Model 3: Linear Regression ───────────────────────────────────────────────

def linear_regression_baseline(splits, scaler):
    tr  = splits["train"]
    va  = splits["val"]
    te  = splits["test"]

    X_tr = flatten(tr["ohlcv"], tr["patterns"])
    X_va = flatten(va["ohlcv"], va["patterns"])
    X_te = flatten(te["ohlcv"], te["patterns"])

    # Combine train + val for final fit
    X_fit = np.vstack([X_tr, X_va])
    y_fit = np.concatenate([tr["price"], va["price"]])

    model = LinearRegression()
    model.fit(X_fit, y_fit)

    pred_sc   = model.predict(X_te)
    true_real = scaler.inverse_close(te["price"])
    pred_real = scaler.inverse_close(pred_sc)
    return true_real, pred_real


# ── Model 4: Random Forest ────────────────────────────────────────────────────

def random_forest_baseline(splits, scaler):
    print("  [Random Forest] Training … ")
    tr = splits["train"]
    va = splits["val"]
    te = splits["test"]

    X_fit = np.vstack([flatten(tr["ohlcv"], tr["patterns"]),
                       flatten(va["ohlcv"], va["patterns"])])
    y_fit = np.concatenate([tr["price"], va["price"]])

    model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  n_jobs=-1, random_state=42)
    model.fit(X_fit, y_fit)

    pred_sc   = model.predict(flatten(te["ohlcv"], te["patterns"]))
    true_real = scaler.inverse_close(te["price"])
    pred_real = scaler.inverse_close(pred_sc)
    return true_real, pred_real


# ── Model 5: Gradient Boosting (XGBoost-style) ───────────────────────────────

def gradient_boosting_baseline(splits, scaler):
    print("  [Gradient Boosting] Training … ")
    tr = splits["train"]
    va = splits["val"]
    te = splits["test"]

    X_fit = np.vstack([flatten(tr["ohlcv"], tr["patterns"]),
                       flatten(va["ohlcv"], va["patterns"])])
    y_fit = np.concatenate([tr["price"], va["price"]])

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    model.fit(X_fit, y_fit)

    pred_sc   = model.predict(flatten(te["ohlcv"], te["patterns"]))
    true_real = scaler.inverse_close(te["price"])
    pred_real = scaler.inverse_close(pred_sc)
    return true_real, pred_real


# ── Load saved Hybrid model results ──────────────────────────────────────────

def load_hybrid_results(ticker, splits, scaler):
    """Load the pre-trained Hybrid CNN+LSTM and run it on the test set."""
    import tensorflow as tf

    out   = os.path.join("outputs", ticker)
    r_path = os.path.join(out, "hybrid_reg.keras")
    c_path = os.path.join(out, "hybrid_cls.keras")

    if not os.path.exists(r_path):
        print("  [Hybrid] No saved model found — run train.py first.")
        return None, None

    reg = tf.keras.models.load_model(r_path)
    te  = splits["test"]
    pred_sc   = reg.predict([te["ohlcv"], te["patterns"]], verbose=0).flatten()
    true_real = scaler.inverse_close(te["price"])
    pred_real = scaler.inverse_close(pred_sc)
    return true_real, pred_real


# ── Build results table ───────────────────────────────────────────────────────

def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, (true, pred) in results.items():
        if true is None:
            continue
        rows.append({
            "Model":     name,
            "RMSE":      round(rmse(true, pred),              4),
            "MAE":       round(mae(true, pred),               4),
            "Dir Acc %": round(direction_accuracy(true, pred) * 100, 2),
        })
    df = pd.DataFrame(rows).sort_values("RMSE")
    df.index = range(1, len(df) + 1)
    return df


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(table: pd.DataFrame, results: dict,
                    dates, out_dir: str):

    # ── Bar charts ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(table)))

    for ax, metric, ascending in zip(axes,
                                     ["RMSE", "MAE", "Dir Acc %"],
                                     [True,   True,  False]):
        sorted_t = table.sort_values(metric, ascending=ascending)
        colors   = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(sorted_t))) \
                   if ascending else \
                   plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(sorted_t)))
        bars = ax.barh(sorted_t["Model"], sorted_t[metric],
                       color=colors, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.3)
        # Highlight the Hybrid model bar
        for bar, label in zip(bars, sorted_t["Model"]):
            if "Hybrid" in label:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.5)

    fig.suptitle("Model Comparison – Test Set", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    bar_path = os.path.join(out_dir, "comparison_bars.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Saved bar chart → {bar_path}")

    # ── Price prediction overlay ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    true_arr  = list(results.values())[0][0]
    ax2.plot(range(len(true_arr)), true_arr,
             label="Actual", color="black", linewidth=2, zorder=10)

    colors_line = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    for i, (name, (true, pred)) in enumerate(results.items()):
        if pred is None or name == "Actual":
            continue
        ax2.plot(range(len(pred)), pred,
                 label=name, alpha=0.7,
                 linewidth=2 if "Hybrid" in name else 1,
                 linestyle="-" if "Hybrid" in name else "--",
                 color=colors_line[i % len(colors_line)])

    ax2.set_title("All Models – Actual vs Predicted (Test Set)", fontsize=13)
    ax2.set_xlabel("Test Day Index")
    ax2.set_ylabel("Price (USD)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    overlay_path = os.path.join(out_dir, "comparison_overlay.png")
    fig2.savefig(overlay_path, dpi=150)
    plt.close(fig2)
    print(f"[compare] Saved overlay chart → {overlay_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(ticker: str = "AAPL", window: int = WINDOW_SIZE):

    out_dir = os.path.join("outputs", ticker)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Model Comparison Pipeline  |  {ticker}")
    print(f"{'#'*60}\n")

    # ── Load data & preprocess ────────────────────────────────────────────────
    df = download_ticker(ticker)

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        print("[compare] Loading saved scaler …")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        splits, _ = prepare_data(df, window=window)
        # Re-fit scaler on same data (consistent)
    else:
        splits, scaler = prepare_data(df, window=window)

    # Raw close prices for ARIMA (not scaled)
    n_train = len(splits["train"]["price"]) + window
    n_val   = len(splits["val"]["price"])
    n_test  = len(splits["test"]["price"])

    close_all   = df["Close"].values
    close_train = close_all[:n_train + n_val]
    close_test  = close_all[n_train + n_val : n_train + n_val + n_test]
    test_dates  = splits["test"]["dates"]

    # ── Run each model ────────────────────────────────────────────────────────
    results = {}

    print("\n[1/6] Naïve Baseline …")
    true_r, pred_naive = naive_baseline(scaler, splits)
    results["Naïve (yesterday)"] = (true_r, pred_naive)

    print("\n[2/6] ARIMA(5,1,0) …")
    try:
        pred_arima = arima_baseline(close_train, close_test)
        results["ARIMA(5,1,0)"] = (close_test, pred_arima)
    except Exception as e:
        print(f"  ARIMA failed: {e}")

    print("\n[3/6] Linear Regression …")
    true_r, pred_lr = linear_regression_baseline(splits, scaler)
    results["Linear Regression"] = (true_r, pred_lr)

    print("\n[4/6] Random Forest …")
    true_r, pred_rf = random_forest_baseline(splits, scaler)
    results["Random Forest"] = (true_r, pred_rf)

    print("\n[5/6] Gradient Boosting …")
    true_r, pred_gb = gradient_boosting_baseline(splits, scaler)
    results["Gradient Boosting"] = (true_r, pred_gb)

    print("\n[6/6] Hybrid CNN+LSTM (your model) …")
    true_r, pred_hybrid = load_hybrid_results(ticker, splits, scaler)
    if pred_hybrid is not None:
        results["★ Hybrid CNN+LSTM"] = (true_r, pred_hybrid)

    # ── Build & print table ───────────────────────────────────────────────────
    table = build_table(results)

    print(f"\n{'='*60}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(table.to_string(index=True))

    # ── Save table ────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "comparison_table.csv")
    table.to_csv(csv_path)
    print(f"\n[compare] Saved table → {csv_path}")

    json_path = os.path.join(out_dir, "comparison_metrics.json")
    table.to_json(json_path, orient="records", indent=2)
    print(f"[compare] Saved JSON  → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(table, results, test_dates, out_dir)

    # ── Improvement summary ───────────────────────────────────────────────────
    if "★ Hybrid CNN+LSTM" in results and "Naïve (yesterday)" in results:
        hybrid_rmse = table.loc[table["Model"] == "★ Hybrid CNN+LSTM", "RMSE"].values[0]
        naive_rmse  = table.loc[table["Model"] == "Naïve (yesterday)",  "RMSE"].values[0]
        improvement = (naive_rmse - hybrid_rmse) / naive_rmse * 100
        print(f"\n🏆 Your Hybrid CNN+LSTM improves RMSE by "
              f"{improvement:.1f}% over the Naïve baseline.")

    print(f"\nAll outputs saved to: {out_dir}/")
    return table


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()
    main(ticker=args.ticker, window=args.window)
