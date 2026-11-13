"""
models.py
=========
Model architecture definitions for:
  1. LSTM-Q: Long Short-Term Memory quantile regression network
  2. CNN-BiGRU-Q: Dilated/Causal CNN + Bidirectional GRU quantile network
  3. DTAQ: Dilated Temporal Convolutional Network + Bahdanau Temporal Attention + Non-crossing Quantiles

All models output 7 ordered quantile forecasts: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95].
Quantile crossing is eliminated by design using the NonCrossingQuantileHead.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import QUANTILES
from common_tf import NonCrossingQuantileHead, BahdanauAttention, pinball_avg_loss


def build_lstm_q(input_shape, lr=1e-3, steps_per_epoch=100):
    """
    Build LSTM-Q model.
    Parameters:
      C1/C2 (26 feats): 25,607 params
      C3 (10 feats):    21,511 params
    """
    inp = keras.Input(shape=input_shape, name="sequence_input")
    x   = layers.LSTM(64, name="lstm")(inp)
    x   = layers.Dropout(0.1, name="dropout")(x)
    x   = layers.Dense(32, activation="relu", name="dense")(x)
    out = NonCrossingQuantileHead(QUANTILES, name="quantile_head")(x)

    model = keras.Model(inp, out, name="LSTM_Q")
    sched = keras.optimizers.schedules.CosineDecayRestarts(
        lr, first_decay_steps=10 * steps_per_epoch, alpha=1e-5
    )
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=pinball_avg_loss(QUANTILES))
    return model


def build_cnn_bigru_q(input_shape, lr=1e-3, steps_per_epoch=100):
    """
    Build CNN-BiGRU-Q model.
    Parameters:
      C1/C2 (26 feats): 72,199 params
      C3 (10 feats):    69,127 params
    """
    inp = keras.Input(shape=input_shape, name="sequence_input")
    x   = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1")(inp)
    x   = layers.BatchNormalization(name="bn1")(x)
    x   = layers.Dropout(0.1, name="conv1_drop")(x)
    x   = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x   = layers.BatchNormalization(name="bn2")(x)
    x   = layers.Dropout(0.1, name="conv2_drop")(x)
    x   = layers.Bidirectional(layers.GRU(64), name="bigru")(x)
    x   = layers.Dropout(0.1, name="bigru_drop")(x)
    x   = layers.Dense(32, activation="relu", name="dense")(x)
    out = NonCrossingQuantileHead(QUANTILES, name="quantile_head")(x)

    model = keras.Model(inp, out, name="CNN_BiGRU_Q")
    sched = keras.optimizers.schedules.CosineDecayRestarts(
        lr, first_decay_steps=10 * steps_per_epoch, alpha=1e-5
    )
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=pinball_avg_loss(QUANTILES))
    return model


def residual_tcn_block(x, filters=64, kernel_size=3, dilation=1, dropout=0.1, block_id=1):
    """Residual causal TCN block with 2 Conv1D + BN + ReLU + Dropout layers."""
    shortcut = x
    y = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation, name=f"tcn_b{block_id}_conv1")(x)
    y = layers.BatchNormalization(name=f"tcn_b{block_id}_bn1")(y)
    y = layers.Activation("relu", name=f"tcn_b{block_id}_act1")(y)
    y = layers.Dropout(dropout, name=f"tcn_b{block_id}_drop1")(y)

    y = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation, name=f"tcn_b{block_id}_conv2")(y)
    y = layers.BatchNormalization(name=f"tcn_b{block_id}_bn2")(y)

    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same",
                                 name=f"tcn_b{block_id}_projection")(shortcut)

    y = layers.Add(name=f"tcn_b{block_id}_add")([shortcut, y])
    return layers.Activation("relu", name=f"tcn_b{block_id}_act2")(y)


def build_dtaq(input_shape, filters=64, dilations=None, lr=1e-3, steps_per_epoch=100):
    """
    Build proposed DTAQ model (Dilated TCN + Bahdanau Attention + Non-crossing Quantiles).
    Theoretical Receptive Field = 1 + 2 * (3-1) * sum([1, 2, 4, 8, 16]) = 125 steps.
    Parameters:
      C1/C2 (26 feats): 127,207 params
      C3 (10 feats):    123,111 params
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16]

    inp = keras.Input(shape=input_shape, name="sequence_input")
    x   = layers.Conv1D(filters, 1, padding="same", name="input_projection")(inp)

    for block_id, dilation in enumerate(dilations, start=1):
        x = residual_tcn_block(
            x, filters=filters, kernel_size=3, dilation=dilation,
            dropout=0.1, block_id=block_id
        )

    context, attention_weights = BahdanauAttention(units=32, name="bahdanau_attention")(x)
    z   = layers.Dense(64, activation="relu", name="context_projection")(context)
    z   = layers.Dropout(0.1, name="dense_dropout")(z)
    out = NonCrossingQuantileHead(QUANTILES, name="quantile_head")(z)

    model = keras.Model(inp, out, name="DTAQ")
    sched = keras.optimizers.schedules.CosineDecayRestarts(
        lr, first_decay_steps=10 * steps_per_epoch, alpha=1e-5
    )
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=pinball_avg_loss(QUANTILES))

    attention_model = keras.Model(inp, attention_weights, name="DTAQ_Attention")
    return model, attention_model
