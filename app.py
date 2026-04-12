"""
app.py  —  Streamlit app. Clean, simple, no clutter.

Run:  streamlit run app.py
"""

import os, json, pickle, glob
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from data_loader       import download_ticker, TICKERS
from preprocessing     import build_feature_frame, DataScaler, WINDOW_SIZE
from pattern_detection import describe_latest_patterns, PATTERN_NAMES, detect_all_patterns

st.set_page_config(page_title="📈 Stock Predictor", layout="wide",
                   initial_sidebar_state="expanded")

OUTPUT_ROOT = "outputs"

# ── Data loading (auto-refreshes every 6 h) ───────────────────────────────────

@st.cache_data(ttl=21600, show_spinner="Fetching latest market data …")
def load_data(ticker: str) -> pd.DataFrame:
    return download_ticker(ticker, end=None)   # end=None → today


@st.cache_resource(show_spinner="Loading model …")
def load_models(ticker: str):
    import tensorflow as tf
    out = os.path.join(OUTPUT_ROOT, ticker)
    rp  = os.path.join(out, "hybrid_reg.keras")
    cp  = os.path.join(out, "hybrid_cls.keras")
    if os.path.exists(rp) and os.path.exists(cp):
        return tf.keras.models.load_model(rp), tf.keras.models.load_model(cp)
    return None, None


def load_meta(ticker: str) -> dict:
    path = os.path.join(OUTPUT_ROOT, ticker, "meta.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def load_metrics(ticker: str) -> dict:
    path = os.path.join(OUTPUT_ROOT, ticker, "metrics.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def load_scaler(ticker: str):
    path = os.path.join(OUTPUT_ROOT, ticker, "scaler.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


# ── Helpers ───────────────────────────────────────────────────────────────────

PATTERN_DESC = {
    "Doji":             "⚖️  Indecision — open ≈ close",
    "Hammer":           "🔨  Bullish reversal — long lower shadow",
    "ShootingStar":     "🌠  Bearish reversal — long upper shadow",
    "BullishEngulfing": "🟢  Strong bullish — green engulfs prior red",
    "BearishEngulfing": "🔴  Strong bearish — red engulfs prior green",
    "MorningStar":      "🌅  3-candle bullish reversal",
    "EveningStar":      "🌆  3-candle bearish reversal",
}


def candlestick_chart(df: pd.DataFrame, n: int = 90) -> go.Figure:
    r = df.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=r.index, open=r["Open"], high=r["High"],
        low=r["Low"], close=r["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="OHLC"))
    fig.update_layout(
        xaxis_rangeslider_visible=False, template="plotly_dark",
        height=380, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
    return fig


def prediction_chart(dates, true_p, pred_p, ticker) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=true_p, name="Actual",
                             line=dict(color="#ecf0f1", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=pred_p, name="Predicted",
                             line=dict(color="#3498db", width=1.5, dash="dot")))
    fig.update_layout(
        title=f"{ticker} — Actual vs Predicted (test set)",
        template="plotly_dark", height=360,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10,r=10,t=50,b=10))
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Settings")
    ticker     = st.sidebar.selectbox("Stock", TICKERS)
    n_chart    = st.sidebar.slider("Chart history (days)", 30, 365, 90)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh data"):
        for f in glob.glob("data/*.csv"): os.remove(f)
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(f"Auto-updates every 6 h · {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.sidebar.markdown("---")
    st.sidebar.info("**Train:**\n```\npython train.py --ticker AAPL\n```")

    # ── Load ──────────────────────────────────────────────────────────────────
    df       = load_data(ticker)
    reg, cls = load_models(ticker)
    meta     = load_meta(ticker)
    scaler   = load_scaler(ticker)

    st.title(f"📈 {ticker} — Stock Market Predictor")
    st.caption(f"Data: {df.index[0].date()} → **{df.index[-1].date()}**  ({len(df)} trading days)")

    # ── Top metrics row ───────────────────────────────────────────────────────
    latest, prev = df.iloc[-1], df.iloc[-2]
    chg  = latest["Close"] - prev["Close"]
    pct  = chg / prev["Close"] * 100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Close",  f"${latest['Close']:.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    c2.metric("High",   f"${latest['High']:.2f}")
    c3.metric("Low",    f"${latest['Low']:.2f}")
    c4.metric("Volume", f"{int(latest['Volume']):,}")

    st.plotly_chart(candlestick_chart(df, n_chart), use_container_width=True)

    st.markdown("---")

    # ── Prediction section ────────────────────────────────────────────────────
    st.subheader("🔮 Next-Day Prediction")

    if reg is None:
        st.warning(
            f"No trained model found for **{ticker}**.  \n"
            f"Run:  `python train.py --ticker {ticker} --epochs 100`"
        )
    else:
        # Use window from training metadata
        window    = meta.get("window", WINDOW_SIZE)
        n_features= meta.get("n_features", None)

        # Build feature frame from all available (fresh) data
        feat        = build_feature_frame(df)
        feat_scaled = scaler.transform(feat)
        X_live      = feat_scaled.values[-window:].reshape(1, window, -1)

        current_close = float(df["Close"].iloc[-1])

        # Predict return → reconstruct price → direction from sign
        pred_return = float(reg.predict(X_live, verbose=0).flatten()[0])
        pred_price  = current_close * (1 + pred_return)
        direction   = "Up ↑" if pred_return > 0 else "Down ↓"
        confidence  = abs(pred_return) * 100      # return magnitude as signal
        dir_color   = "#2ecc71" if pred_return > 0 else "#e74c3c"

        p1, p2, p3 = st.columns(3)
        p1.metric("Current Close",        f"${current_close:.2f}")
        p2.metric("Predicted Next Close", f"${pred_price:.2f}",
                  f"{pred_price - current_close:+.2f} ({pred_return*100:+.2f}%)")
        p3.metric("Predicted Direction",  direction,
                  f"Expected move: {pred_return*100:+.2f}%")

        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:8px;"
            f"background:{dir_color}22;border:1px solid {dir_color};margin:8px 0'>"
            f"<span style='color:{dir_color};font-size:1.4rem;font-weight:bold'>"
            f"Tomorrow's forecast: {direction}</span></div>",
            unsafe_allow_html=True)

        # Test-set chart
        st.markdown("#### Test set: Actual vs Predicted")
        te = None
        try:
            from preprocessing import prepare_data
            splits, _, _ = prepare_data(df, window=window)
            te = splits["test"]
            pred_rets   = reg.predict(te["X"], verbose=0).flatten()
            true_prices = te["last_closes"] * (1 + te["y_return"])
            pred_prices = te["last_closes"] * (1 + pred_rets)
            st.plotly_chart(
                prediction_chart(te["dates"], true_prices, pred_prices, ticker),
                use_container_width=True)
        except Exception as e:
            st.info(f"Could not render test chart: {e}")

    st.markdown("---")

    # ── Pattern section ───────────────────────────────────────────────────────
    st.subheader("🕯️ Candlestick Patterns — Latest Day")

    info   = describe_latest_patterns(df)
    fired  = [p for p in PATTERN_NAMES if info["all"].get(p, 0) == 1]

    if fired:
        st.success(f"**{len(fired)} pattern(s) on {info['date']}:** " + " · ".join(fired))
    else:
        st.info(f"No textbook patterns detected on {info['date']} — normal on most trading days.")

    cols = st.columns(2)
    for i, name in enumerate(PATTERN_NAMES):
        is_fired = info["all"].get(name, 0) == 1
        icon     = "✅" if is_fired else "⬜"
        desc     = PATTERN_DESC.get(name, name)
        cols[i % 2].markdown(f"{icon} **{name}**  \n<small>{desc}</small>",
                             unsafe_allow_html=True)

    # Pattern frequency bar
    st.markdown("#### Pattern frequency (full dataset)")
    pat_df = detect_all_patterns(df)
    freq   = pat_df.sum().sort_values(ascending=True)
    fig_f  = go.Figure(go.Bar(
        x=freq.values, y=freq.index, orientation="h",
        marker_color="#3498db", text=freq.values, textposition="outside"))
    fig_f.update_layout(template="plotly_dark", height=300,
                        margin=dict(l=10,r=10,t=10,b=10),
                        xaxis_title="Occurrences")
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")

    # ── Model comparison ──────────────────────────────────────────────────────
    st.subheader("📋 Model Comparison (Test Set)")
    metrics = load_metrics(ticker)
    if metrics:
        mdf = pd.DataFrame(metrics).T.reset_index()
        mdf.columns = ["Model","RMSE ($)","MAE ($)","Accuracy"]
        mdf = mdf.sort_values("RMSE ($)")
        st.dataframe(
            mdf.style
               .highlight_min(subset=["RMSE ($)","MAE ($)"], color="#2ecc7133")
               .highlight_max(subset=["Accuracy"],            color="#2ecc7133")
               .format({"RMSE ($)":"{:.2f}","MAE ($)":"{:.2f}","Accuracy":"{:.3f}"}),
            use_container_width=True, hide_index=True)

        img = os.path.join(OUTPUT_ROOT, ticker, "model_comparison.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
    else:
        st.info(f"Run `python train.py --ticker {ticker}` to generate metrics.")


if __name__ == "__main__":
    main()
