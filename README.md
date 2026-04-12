# 📈 Stock Market Prediction — Hybrid CNN + LSTM

> End-to-end ML pipeline with **explicit candlestick pattern detection**, a hybrid
> **CNN + LSTM** model, baseline comparisons, and a **Streamlit** web app.

---

## 🗂️ Project Structure

```
stock_prediction/
├── data_loader.py          # Downloads & caches OHLCV data via yfinance
├── pattern_detection.py    # Rule-based candlestick pattern detectors  ← KEY
├── preprocessing.py        # Feature engineering, normalisation, windows
├── models.py               # All Keras model architectures
├── train.py                # Training + evaluation + plots
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── Stock_Prediction_Colab.ipynb   # Ready-to-run Colab notebook
└── outputs/
    └── <TICKER>/
        ├── hybrid_reg.keras
        ├── hybrid_cls.keras
        ├── lstm_reg.keras   …
        ├── scaler.pkl
        ├── metrics.json
        ├── actual_vs_predicted.png
        ├── confusion_matrix.png
        └── model_comparison.png
```

---

## 🕯️ Candlestick Patterns Detected

| Pattern | Type | Rule summary |
|---|---|---|
| **Doji** | Reversal | Body < 5 % of full range |
| **Hammer** | Bullish reversal | Small body, lower shadow ≥ 2× body |
| **Shooting Star** | Bearish reversal | Small body, upper shadow ≥ 2× body |
| **Bullish Engulfing** | Bullish reversal | Green candle engulfs prior red |
| **Bearish Engulfing** | Bearish reversal | Red candle engulfs prior green |
| **Morning Star** | Bullish reversal | 3-candle: big red → doji → big green |
| **Evening Star** | Bearish reversal | 3-candle: big green → doji → big red |

Each pattern is encoded as a **binary flag** (1 / 0) and stacked as the *pattern branch* input to the CNN.

---

## 🧠 Model Architecture

```
OHLCV Input (30, 8)            Pattern Input (30, 7)
      │                                │
  LSTM(64, seq=True)           Reshape → (30, 7, 1)
  Dropout(0.3)                 Conv2D(32, 3×3) + ReLU
  LSTM(32)                     MaxPool(2×2)
  Dropout(0.3)                 Conv2D(64, 3×3) + ReLU
  BatchNorm                    GlobalAvgPool
      │                        Dropout(0.3)
      └──────── Concat ─────────────┘
                   │
              Dense(64) + ReLU
              Dropout(0.3)
              BatchNorm
                   │
       ┌───────────┴────────────┐
  Dense(1, linear)        Dense(1, sigmoid)
  Regression head         Classification head
  (next-day price)        (Up=1 / Down=0)
```

---

## ⚙️ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (example: AAPL, 100 epochs)

```bash
python train.py --ticker AAPL --window 30 --epochs 100
```

Optional args: `--ticker`, `--window`, `--epochs`, `--batch`

### 3. Launch the web app

```bash
streamlit run app.py
```

---

## 🚀 Google Colab

Open `Stock_Prediction_Colab.ipynb` in Colab (GPU runtime recommended):

1. Install deps (Cell 1)
2. Upload project files (Cell 2)
3. Run data checks + pattern analysis (Cells 3–4)
4. Train models (Cell 6)
5. Launch Streamlit via ngrok (Cell 8)

---

## 📊 Feature Engineering Details

| Feature group | Columns | Notes |
|---|---|---|
| Raw OHLCV | Open, High, Low, Close, Volume | Min-max scaled to [0,1] |
| Derived | Return, HL_Range, Body | Daily % changes |
| Patterns | 7 binary flags | Not scaled (already 0/1) |

Sliding window: **last 30 days → predict day 31**

---

## 📈 Evaluation Metrics

| Metric | Task |
|---|---|
| RMSE / MAE | Regression (price) |
| Accuracy | Classification (direction) |
| Confusion Matrix | Hybrid model direction prediction |

---

## 🌐 Web App Features

- **Stock selector** dropdown (AAPL, MSFT, TSLA, AMZN, GOOGL)
- **Candlestick chart** (adjustable lookback)
- **Next-day prediction**: price + direction with confidence
- **Pattern flags** for the latest trading day (✅ / ❌)
- **Pattern explainability**: overlay of fired patterns on price chart
- **Model comparison** metrics and charts
- **Adjustable window size** slider

---

## 📌 Notes

- No sentiment/news data used — pure price action only
- Models are simple and readable — no black-box magic
- Pattern detection uses standard OHLC rules (no TA-Lib dependency)
- All outputs are reproducible from `train.py`
