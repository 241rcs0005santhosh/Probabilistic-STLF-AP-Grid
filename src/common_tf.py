"""
common_tf.py  --  TensorFlow helpers for DTAQ, LSTM-Q and CNN-BiGRU-Q
Depends on common.py for all numpy/pandas helpers and constants.
"""

import os
import sys
import random
import time
import json
import zipfile
import copy
import gc

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import all helpers from common.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# Try to enable deterministic ops (TF >= 2.9)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

for _g in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass


def reseed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------------ custom layers
class NonCrossingQuantileHead(layers.Layer):
    """
    Non-crossing quantile output head .
    Guarantees quantile ordering via softplus increments from the median.
    Used by all three models: LSTM-Q, CNN-BiGRU-Q, and DTAQ.
    """
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = list(quantiles)
        self.n_q       = len(self.quantiles)
        self.m_idx     = self.quantiles.index(0.50)
        self.dense     = layers.Dense(self.n_q)

    def call(self, x):
        raw  = self.dense(x)
        med  = raw[:, self.m_idx:self.m_idx + 1]
        outs = [None] * self.n_q
        outs[self.m_idx] = med
        cum = med
        for i in range(self.m_idx + 1, self.n_q):
            cum     = cum + tf.nn.softplus(raw[:, i:i + 1])
            outs[i] = cum
        cum = med
        for i in range(self.m_idx - 1, -1, -1):
            cum     = cum - tf.nn.softplus(raw[:, i:i + 1])
            outs[i] = cum
        return tf.concat(outs, axis=1)

    def get_config(self):
        c = super().get_config()
        c["quantiles"] = self.quantiles
        return c


class BahdanauAttention(layers.Layer):
    """
    Bahdanau-style temporal attention over TCN hidden states .
    Returns (context_vector, attention_weights).
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W_h   = layers.Dense(units, use_bias=False)
        self.v     = layers.Dense(1,     use_bias=False)
        self.units = units

    def call(self, h):
        w       = tf.nn.softmax(self.v(tf.nn.tanh(self.W_h(h))), axis=1)  # (B, L, 1)
        context = tf.reduce_sum(w * h, axis=1)
        return context, w

    def get_config(self):
        c = super().get_config()
        c["units"] = self.units
        return c


class PerturbableAttention(BahdanauAttention):
    """
    Same weights and structure as BahdanauAttention; a global tf.Variable
    switch changes how the weights are used at inference.

    Modes:
        0 -- original (unchanged weights)
        1 -- uniform attention weights
        2 -- shuffled across time (per window)
        3 -- mask top-k attended steps
        4 -- mask the last step (15 min before target)
    """
    MODE = None
    K    = None

    @classmethod
    def switches(cls):
        if cls.MODE is None:
            cls.MODE = tf.Variable(0,      trainable=False, dtype=tf.int32)
            cls.K    = tf.Variable(1,      trainable=False, dtype=tf.int32)
        return cls.MODE, cls.K

    def call(self, h):
        w       = tf.nn.softmax(self.v(tf.nn.tanh(self.W_h(h))), axis=1)
        mode, k = self.switches()
        L       = tf.shape(w)[1]
        uniform = tf.ones_like(w) / tf.cast(L, w.dtype)
        shuf    = tf.transpose(tf.random.shuffle(tf.transpose(w, [1, 0, 2])), [1, 0, 2])
        vals, _ = tf.math.top_k(tf.squeeze(w, -1), k=tf.minimum(k, L))
        thr     = vals[:, -1][:, None, None]
        mk      = tf.where(w >= thr, tf.zeros_like(w), w)
        mk      = mk / (tf.reduce_sum(mk, axis=1, keepdims=True) + 1e-9)
        last    = tf.concat([w[:, :-1], tf.zeros_like(w[:, -1:])], axis=1)
        last    = last / (tf.reduce_sum(last, axis=1, keepdims=True) + 1e-9)
        w2 = tf.where(mode == 0, w,
             tf.where(mode == 1, uniform,
             tf.where(mode == 2, shuf,
             tf.where(mode == 3, mk, last))))
        return tf.reduce_sum(w2 * h, axis=1), w


PerturbableAttention.switches()

CUSTOM_OBJS = {
    "NonCrossingQuantileHead": NonCrossingQuantileHead,
    "BahdanauAttention":       BahdanauAttention,
}


# ------------------------------------------------------------------ loss
def pinball_avg_loss(quantiles=QUANTILES):
    """Average 7-quantile pinball loss used for all models."""
    qt = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        e = tf.expand_dims(y_true, -1) - y_pred
        return tf.reduce_mean(
            tf.reduce_mean(tf.maximum(qt * e, (qt - 1.0) * e), axis=-1)
        )
    return loss


# ------------------------------------------------------------------ checkpoint loading / rebuilding
def load_ckpt(path, perturbable=False):
    """Load a .keras checkpoint with custom objects."""
    co = dict(CUSTOM_OBJS)
    if perturbable:
        co["BahdanauAttention"] = PerturbableAttention
    return keras.models.load_model(path, custom_objects=co, compile=False)


def rebuild_arch(path, lb, nf, seed=None):
    """
    Fresh (randomly initialised) copy of the architecture stored in a checkpoint,
    with a new (lb, nf) input.  Used for component ablation variants.
    """
    cfg   = copy.deepcopy(read_cfg(path))
    inner = cfg["config"]
    for l in inner.get("layers", []):
        l.pop("build_config", None)
        c = l.get("config", {})
        c.pop("build_config", None)
        if l["class_name"] == "InputLayer":
            c["batch_shape" if "batch_shape" in c else "batch_input_shape"] = [None, lb, nf]
        elif "batch_input_shape" in c:
            c["batch_input_shape"] = [None, lb, nf]
    inner.pop("build_input_shape", None)
    if seed is not None:
        reseed(seed)
    cls = keras.Sequential if cfg["class_name"] == "Sequential" else keras.Model
    m   = cls.from_config(inner, custom_objects=CUSTOM_OBJS)
    if not m.built:
        m.build((None, lb, nf))
    return m


def find_layer(model, cls):
    return next((l for l in model.layers if isinstance(l, cls)), None)


# ------------------------------------------------------------------ training / prediction
class _Tick(keras.callbacks.Callback):
    def __init__(self, tag, every=5):
        super().__init__()
        self.tag   = tag
        self.every = every

    def on_epoch_end(self, ep, logs=None):
        if (ep + 1) % self.every == 0 or ep == 0:
            print(f"    [{self.tag}] epoch {ep + 1}  "
                  f"loss={logs['loss']:.4f}  val_loss={logs['val_loss']:.4f}", flush=True)


def train_model(model, Xtr, ytr, Xvl, yvl, arch, seed, tag="", batch=256):
    """Train a model using cosine-decay LR and early stopping."""
    steps = int(np.ceil(len(Xtr) / batch))
    lr    = keras.optimizers.schedules.CosineDecayRestarts(
        1e-3, first_decay_steps=10 * steps, alpha=1e-5
    )
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=pinball_avg_loss())
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE[arch],
            restore_best_weights=True,
            start_from_epoch=min(5, MAX_EPOCHS[arch] - 1), verbose=0
        ),
        _Tick(tag)
    ]
    reseed(seed)
    t0 = time.time()
    h  = model.fit(
        Xtr, ytr, validation_data=(Xvl, yvl),
        epochs=MAX_EPOCHS[arch], batch_size=batch,
        callbacks=cb, verbose=0
    )
    return dict(
        epochs=len(h.history["loss"]),
        seconds=round(time.time() - t0, 1),
        best_val=float(np.min(h.history["val_loss"])),
        params=int(model.count_params())
    )


def predict_q(model, X, bs=1024):
    return np.asarray(model.predict(X, batch_size=bs, verbose=0), dtype=np.float64)


def q_to_mw(prep, yq_scaled):
    return np.column_stack([inv(prep, yq_scaled[:, i]) for i in range(yq_scaled.shape[1])])


def export_preds(model, prep, tag, extra=None):
    """
    Predict calibration + test windows, convert to MW, compute q_hat on RAW
    quantiles, and save to .npz cache file.
    """
    ic, it = prep.idx["cal"], prep.idx["test"]
    yq_c   = q_to_mw(prep, predict_q(model, prep.X[ic]))
    yq_t   = q_to_mw(prep, predict_q(model, prep.X[it]))
    y_c    = inv(prep, prep.y[ic])
    y_t    = inv(prep, prep.y[it])
    qhat   = cqr_qhat(cqr_scores(y_c, yq_c))
    save_preds(
        os.path.join(D_CACHE, f"{tag}.npz"),
        ts_cal=prep.ts[ic], y_cal=y_c, yq_cal=yq_c,
        ts_test=prep.ts[it], y_test=y_t, yq_test=yq_t,
        q_hat=np.array([qhat]),
        **(extra or {})
    )
    return y_t, yq_t, qhat


def summarize(tag, y_t, yq_t, qhat, info=None):
    row = dict(tag=tag, q_hat=qhat, **point_metrics(y_t, yq_t[:, MED]))
    row.update({f"{k}_raw": v for k, v in prob_metrics(y_t, yq_t).items()})
    row.update({f"{k}_cqr": v for k, v in prob_metrics(y_t, shift_outer(yq_t, qhat)).items()})
    row.update(info or {})
    return row


# ------------------------------------------------------------------ permutation importance
def perm_importance(model, X, y_mw, prep, K=10, seed=SEED, bs=2048):
    rng  = np.random.default_rng(seed)
    F    = X.shape[2]
    base = float(np.sqrt(np.mean(
        (y_mw - inv(prep, predict_q(model, X, bs)[:, MED])) ** 2
    )))
    out  = np.zeros((K, F))
    for k in range(K):
        for f in range(F):
            Xp         = X.copy()
            Xp[:, :, f] = X[rng.permutation(len(X)), :, f]
            out[k, f]  = np.sqrt(np.mean(
                (y_mw - inv(prep, predict_q(model, Xp, bs)[:, MED])) ** 2
            )) - base
    return base, out


# ------------------------------------------------------------------ component-study variants (exact DTAQ backbone)
def backbone_params(model):
    return int(keras.Model(
        model.input,
        find_layer(model, BahdanauAttention).input
    ).count_params())


def match_units(backbone_n, channels, target):
    best = min(
        range(8, 2049),
        key=lambda u: abs(backbone_n + (channels + 1) * u + (u + 1) * Q - target)
    )
    return best, backbone_n + (channels + 1) * best + (best + 1) * Q


def build_variant(dtaq_ckpt, lb, nf, variant, seed):
    """
    Build ablation variants sharing the EXACT DTAQ backbone (fresh weights).
    Real checkpoint head: BahdanauAttention(units=32) -> Dense(64, relu)
                          -> Dropout(0.1) -> NonCrossingQuantileHead.

    Variants:
        "attention" -- full DTAQ (attention pooling)
        "last"      -- last-step pooling
        "gap"       -- global average pooling
        "gap_matched" -- global average pooling, matched parameter count
    """
    full = rebuild_arch(dtaq_ckpt, lb, nf, seed)
    if variant == "attention":
        return full
    h = find_layer(full, BahdanauAttention).input
    C = int(h.shape[-1])
    if variant == "last":
        x = layers.Lambda(lambda t: t[:, -1, :], name="read_last")(h)
        u = 64
    else:
        x = layers.GlobalAveragePooling1D(name="read_gap")(h)
        u = 64
        if variant == "gap_matched":
            u, _ = match_units(backbone_params(full), C, int(full.count_params()))
    x   = layers.Dense(u, activation="relu", name="proj_var")(x)
    x   = layers.Dropout(0.1, name="proj_drop_var")(x)
    return keras.Model(
        full.input,
        NonCrossingQuantileHead(QUANTILES, name="quantile_head_var")(x),
        name=f"DTAQ_{variant}"
    )


def attention_profile(model, prep, part="test"):
    """
    Mean attention weight per look-back step, overall and by season.
    Index L-1 = 15 min before the target.
    """
    att = find_layer(model, BahdanauAttention)
    wm  = keras.Model(model.input, att.output[1])
    ii  = prep.idx[part]
    w   = np.concatenate([
        wm.predict(prep.X[ii][k:k + 4096], verbose=0)[..., 0]
        for k in range(0, len(ii), 4096)
    ])
    seas = season_of(prep.ts[ii])
    L    = w.shape[1]
    out  = pd.DataFrame(dict(
        step_index=np.arange(L),
        steps_before_target=L - np.arange(L),
        minutes_before_target=(L - np.arange(L)) * 15,
        All=w.mean(0)
    ))
    for s_ in ("Summer", "Monsoon", "Winter"):
        out[s_] = w[seas == s_].mean(0) if (seas == s_).any() else np.nan
    return out
