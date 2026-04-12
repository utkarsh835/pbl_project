"""
train.py  —  trains all models, evaluates on test set, saves outputs.

Usage:
    python train.py --ticker AAPL --window 30 --epochs 100
"""

import os, json, pickle, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, mean_squared_error,
                              mean_absolute_error)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

from data_loader    import download_ticker
from preprocessing  import prepare_data, WINDOW_SIZE
from models         import (build_hybrid_cnn_lstm, build_lstm_only,
                             build_cnn_only, build_mlp)

DEFAULT_TICKER = "AAPL"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH  = 32
OUTPUT_ROOT    = "outputs"


def _out(ticker):
    p = os.path.join(OUTPUT_ROOT, ticker); os.makedirs(p, exist_ok=True); return p


def _callbacks(path, patience=15):
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(path, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                          min_lr=1e-6, verbose=1),
    ]


def reconstruct_price(last_closes, pred_returns):
    """predicted_price = last_close × (1 + predicted_return)"""
    return last_closes * (1 + pred_returns)


def train_one(reg, cls, splits, out_dir, tag, epochs, batch):
    tr, va = splits["train"], splits["val"]
    print(f"\n{'='*55}\n  {tag} — REGRESSION\n{'='*55}")
    reg.fit(tr["X"], tr["y_return"],
            validation_data=(va["X"], va["y_return"]),
            epochs=epochs, batch_size=batch,
            callbacks=_callbacks(os.path.join(out_dir, f"{tag.lower()}_reg.keras")),
            verbose=1)

    print(f"\n{'='*55}\n  {tag} — CLASSIFICATION\n{'='*55}")
    cls.fit(tr["X"], tr["y_direction"],
            validation_data=(va["X"], va["y_direction"]),
            epochs=epochs, batch_size=batch,
            callbacks=_callbacks(os.path.join(out_dir, f"{tag.lower()}_cls.keras")),
            verbose=1)

    te = splits["test"]
    pred_ret = reg.predict(te["X"], verbose=0).flatten()
    pred_dir = cls.predict(te["X"], verbose=0).flatten()
    return pred_ret, pred_dir


def evaluate(results, splits, out_dir):
    te          = splits["test"]
    true_ret    = te["y_return"]
    true_dir    = te["y_direction"].astype(int)
    last_closes = te["last_closes"]
    dates       = te["dates"]
    true_prices = reconstruct_price(last_closes, true_ret)

    metrics = {}
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(dates, true_prices, label="Actual", color="black", lw=2)

    colors = ["steelblue", "tomato", "seagreen", "orange"]
    for i, (tag, (pred_ret, _)) in enumerate(results.items()):
        pred_prices = reconstruct_price(last_closes, pred_ret)
        # Direction derived from return sign — always consistent
        pred_dir    = (pred_ret > 0).astype(int)

        rmse = float(np.sqrt(mean_squared_error(true_prices, pred_prices)))
        mae  = float(mean_absolute_error(true_prices, pred_prices))
        acc  = float(accuracy_score(true_dir, pred_dir))

        metrics[tag] = {"RMSE": round(rmse,2), "MAE": round(mae,2),
                        "Accuracy": round(acc,4)}
        print(f"[{tag:20s}]  RMSE=${rmse:.2f}  MAE=${mae:.2f}  Acc={acc:.3f}")
        ax.plot(dates, pred_prices, label=f"{tag} (RMSE=${rmse:.2f})",
                alpha=0.75, color=colors[i % len(colors)], lw=1.2)

    ax.set_title("Actual vs Predicted Close Price — Test Set")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # Comparison bar chart
    mdf = pd.DataFrame(metrics).T
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax2, col, color in zip(axes, ["RMSE","MAE","Accuracy"],
                                      ["salmon","skyblue","mediumseagreen"]):
        mdf[col].plot(kind="bar", ax=ax2, color=color, edgecolor="black")
        ax2.set_title(col); ax2.tick_params(axis="x", rotation=30)
        ax2.grid(axis="y", alpha=0.3)
    fig2.suptitle("Model Comparison", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150)
    plt.close(fig2)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll saved to {out_dir}/")
    return metrics


def main(ticker=DEFAULT_TICKER, window=WINDOW_SIZE,
         epochs=DEFAULT_EPOCHS, batch=DEFAULT_BATCH):
    out_dir = _out(ticker)
    print(f"\n{'#'*55}\n  {ticker}  |  window={window}  epochs={epochs}\n{'#'*55}")

    df = download_ticker(ticker)
    splits, scaler, n_features = prepare_data(df, window=window)

    # Save scaler + metadata
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"window": window, "n_features": n_features}, f)

    builders = {
        "Hybrid": build_hybrid_cnn_lstm,
        "LSTM":   build_lstm_only,
        "CNN":    build_cnn_only,
        "MLP":    build_mlp,
    }

    results = {}
    for tag, fn in builders.items():
        reg, cls = fn(window=window, n_features=n_features)
        pred_ret, pred_dir = train_one(reg, cls, splits, out_dir, tag, epochs, batch)
        results[tag] = (pred_ret, pred_dir)

    metrics = evaluate(results, splits, out_dir)
    print("\n" + pd.DataFrame(metrics).T.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker",  default=DEFAULT_TICKER)
    p.add_argument("--window",  type=int, default=WINDOW_SIZE)
    p.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    a = p.parse_args()
    main(a.ticker, a.window, a.epochs, a.batch)
