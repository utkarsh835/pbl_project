"""
models.py
=========
Simplified, single-input models.
All features (OHLCV returns + patterns) are concatenated into one
(window, n_features) tensor — cleaner than dual-branch when patterns are sparse.

Hybrid CNN+LSTM:  Conv1D to capture local patterns → LSTM for sequence → Dense
LSTM only:        LSTM → Dense
CNN only:         Conv1D → GlobalAvgPool → Dense
MLP:              Flatten → Dense stack
"""

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

LR      = 3e-4
DROPOUT = 0.3
L2_REG  = 1e-4


def _reg_cls_heads(z, dense_units=64, dropout=DROPOUT, lr=LR, name="Model"):
    """Attach regression + classification output heads, compile, return both models."""
    z = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=l2(L2_REG))(z)
    z = layers.Dropout(dropout)(z)

    # Keep backbone input reference
    inp = z  # will be set correctly by callers

    reg_out = layers.Dense(1, activation="linear",  name="return_output")(z)
    cls_out = layers.Dense(1, activation="sigmoid", name="dir_output")(z)
    return reg_out, cls_out


def build_hybrid_cnn_lstm(window: int, n_features: int):
    """
    Conv1D extracts local temporal patterns (e.g. 3-day candlestick shapes),
    LSTM captures the longer sequence dependency.
    """
    inp = Input(shape=(window, n_features), name="features")

    # Conv1D branch: detect local patterns in the feature sequence
    x = layers.Conv1D(64, kernel_size=3, padding="same",
                      activation="relu", name="conv1")(inp)
    x = layers.Conv1D(32, kernel_size=3, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool")(x)
    x = layers.Dropout(DROPOUT)(x)

    # LSTM on top of conv features
    x = layers.LSTM(64, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm2")(x)
    x = layers.BatchNormalization()(x)

    z = layers.Dense(64, activation="relu")(x)
    z = layers.Dropout(DROPOUT)(z)

    reg_out = layers.Dense(1, activation="linear",  name="return_output")(z)
    cls_out = layers.Dense(1, activation="sigmoid", name="dir_output")(z)

    reg = Model(inp, reg_out, name="Hybrid_Reg")
    cls = Model(inp, cls_out, name="Hybrid_Cls")
    reg.compile(optimizer=Adam(LR), loss="mse",               metrics=["mae"])
    cls.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    return reg, cls


def build_lstm_only(window: int, n_features: int):
    inp = Input(shape=(window, n_features), name="features")
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.LSTM(32, return_sequences=False)(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(DROPOUT)(x)

    reg_out = layers.Dense(1, activation="linear",  name="return_output")(x)
    cls_out = layers.Dense(1, activation="sigmoid", name="dir_output")(x)

    reg = Model(inp, reg_out, name="LSTM_Reg")
    cls = Model(inp, cls_out, name="LSTM_Cls")
    reg.compile(optimizer=Adam(LR), loss="mse",               metrics=["mae"])
    cls.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    return reg, cls


def build_cnn_only(window: int, n_features: int):
    inp = Input(shape=(window, n_features), name="features")
    x   = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x   = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(DROPOUT)(x)

    reg_out = layers.Dense(1, activation="linear",  name="return_output")(x)
    cls_out = layers.Dense(1, activation="sigmoid", name="dir_output")(x)

    reg = Model(inp, reg_out, name="CNN_Reg")
    cls = Model(inp, cls_out, name="CNN_Cls")
    reg.compile(optimizer=Adam(LR), loss="mse",               metrics=["mae"])
    cls.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    return reg, cls


def build_mlp(window: int, n_features: int):
    inp = Input(shape=(window, n_features), name="features")
    x   = layers.Flatten()(inp)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(64,  activation="relu")(x)

    reg_out = layers.Dense(1, activation="linear",  name="return_output")(x)
    cls_out = layers.Dense(1, activation="sigmoid", name="dir_output")(x)

    reg = Model(inp, reg_out, name="MLP_Reg")
    cls = Model(inp, cls_out, name="MLP_Cls")
    reg.compile(optimizer=Adam(LR), loss="mse",               metrics=["mae"])
    cls.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    return reg, cls


if __name__ == "__main__":
    for fn in [build_hybrid_cnn_lstm, build_lstm_only, build_cnn_only, build_mlp]:
        reg, cls = fn(window=30, n_features=17)
        print(f"{reg.name}: {reg.count_params():,} params")
