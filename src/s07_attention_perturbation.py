"""
s07_attention_perturbation.py
=============================
Step 7 -- At-inference attention weight perturbation diagnostics (DTAQ C2).

Purpose:
  Perturbs the softmax attention weights inside the Bahdanau layer of the
  trained DTAQ C2 checkpoint at inference time and measures the effect on RMSE.
  Modes tested:
    0 -- Original attention weights
    1 -- Uniform attention weights across look-back window
    2 -- Attention weights shuffled randomly across time per window
    3 -- Top-3 attended steps masked (zeroed and re-normalized)
    4 -- Last step (15 min before target) masked
  Also extracts the seasonal attention profile and computes profile stability.

Output:
  tables/T_attention_perturbation.csv|docx
  tables/T_attention_stability.csv|docx
  results/attention_profile_C2_DTAQ.csv
  figures/F13_attention_profile_inset.png

"""

import os
import sys
import types
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_tf import *

K_TOP = 3
MODES = {
    0: "original",
    1: "uniform attention weights",
    2: "weights shuffled across time (per window)",
    3: f"top-{K_TOP} attended steps masked",
    4: "last step (15 min before target) masked"
}


def dynamic_perturbed_call(self, h):
    w = tf.nn.softmax(self.v(tf.nn.tanh(self.W_h(h))), axis=1)
    L = tf.shape(w)[1]
    uniform = tf.ones_like(w) / tf.cast(L, w.dtype)
    perm = tf.argsort(tf.random.uniform(tf.shape(w)[:2]), axis=1)
    shuffled = tf.gather(w, perm, batch_dims=1)
    vals, _ = tf.math.top_k(tf.squeeze(w, -1), k=tf.minimum(self.K, L))
    thr = vals[:, -1][:, None, None]
    masked_top = tf.where(w >= thr, tf.zeros_like(w), w)
    masked_top = masked_top / (tf.reduce_sum(masked_top, axis=1, keepdims=True) + 1e-9)
    last = tf.concat([w[:, :-1], tf.zeros_like(w[:, -1:])], axis=1)
    masked_last = last / (tf.reduce_sum(last, axis=1, keepdims=True) + 1e-9)
    w2 = tf.where(self.MODE == 0, w,
         tf.where(self.MODE == 1, uniform,
         tf.where(self.MODE == 2, shuffled,
         tf.where(self.MODE == 3, masked_top, masked_last))))
    return tf.reduce_sum(w2 * h, axis=1), w


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nSet P3_DATA environment variable.")
    ckpt_file = ckpt_path("C2", "DTAQ")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    df, feats, ap = load_frame()
    p = prepare(df, feats, 48, "legacy")
    te = p.idx["test"]
    y_te = inv(p, p.y[te])

    base_model = load_ckpt(ckpt_file)
    reseed(SEED)
    perturbed_model = rebuild_arch(ckpt_file, 48, len(feats), seed=SEED)

    att_layer = find_layer(perturbed_model, BahdanauAttention)
    att_layer.MODE = tf.Variable(0, trainable=False, dtype=tf.int32)
    att_layer.K = tf.Variable(K_TOP, trainable=False, dtype=tf.int32)
    att_layer.call = types.MethodType(dynamic_perturbed_call, att_layer)
    perturbed_model.set_weights(base_model.get_weights())

    rows = []
    B = 200 if QUICK else 2000
    rmse = lambda yy, pp: float(np.sqrt(np.mean((yy - pp) ** 2)))

    med_base = None
    base_rmse = None
    for mode_val, name in MODES.items():
        att_layer.MODE.assign(mode_val)
        yq = q_to_mw(p, predict_q(perturbed_model, p.X[te]))
        if mode_val == 0:
            med_base = yq[:, MED].copy()
            base_rmse = rmse(y_te, med_base)
            d, lo, hi = 0.0, 0.0, 0.0
        else:
            d = rmse(y_te, yq[:, MED]) - base_rmse
            lo, hi = block_boot_ci(
                lambda yy, b_, c_: rmse(yy, c_) - rmse(yy, b_),
                [y_te, med_base, yq[:, MED]], block=96, B=B
            )
        rows.append(dict(
            variant=name, RMSE=rmse(y_te, yq[:, MED]),
            MAE=point_metrics(y_te, yq[:, MED])["MAE"],
            dRMSE_vs_recorded=d, dRMSE_CI_lo=lo, dRMSE_CI_hi=hi
        ))

    T_att = pd.DataFrame(rows)
    write_table(
        "T_attention_perturbation", T_att,
        "At-inference attention weight perturbations on DTAQ (C2) test windows.",
        "Perturbations target the softmax attention weights inside the Bahdanau layer. "
        "Bootstrap CIs (moving block, block=96) assess the significance of the RMSE increase compared to original weights."
    )
    print(T_att.to_string(index=False))

    # Attention profile and stability
    att_layer.MODE.assign(0)
    assert abs(rmse(y_te, med_base) - 173.88) < 0.5, "headline RMSE mismatch"

    wm = keras.Model(perturbed_model.input, att_layer.output[1])
    W = np.concatenate([wm.predict(p.X[te][k:k + 4096], verbose=0)[..., 0] for k in range(0, len(te), 4096)])
    L = W.shape[1]
    seas = season_of(p.ts[te])
    half = np.arange(len(te)) < len(te) // 2

    prof = {
        "All": W.mean(0),
        **{s_: W[seas == s_].mean(0) for s_ in ("Summer", "Monsoon", "Winter")},
        "first_half": W[half].mean(0),
        "second_half": W[~half].mean(0)
    }
    prof_df = pd.DataFrame({
        "step_index": np.arange(L),
        "minutes_before_target": (L - np.arange(L)) * 15,
        **prof
    })
    prof_csv = os.path.join(D_RES, "attention_profile_C2_DTAQ.csv")
    prof_df.to_csv(prof_csv, index=False)

    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    pairs = [("Summer", "Monsoon"), ("Summer", "Winter"), ("Monsoon", "Winter"), ("first_half", "second_half")]
    sim = pd.DataFrame([dict(comparison=f"{a} vs {b}", cosine=cos(prof[a], prof[b])) for a, b in pairs])
    write_table(
        "T_attention_stability", sim,
        "Cosine similarity of mean attention profiles (C2 DTAQ).",
        "Mean attention weight per look-back step, computed on original (unperturbed) weights."
    )
    print(sim.round(4).to_string(index=False))
    print("mean weight on the last 4 steps (last hour):", round(float(prof["All"][-4:].sum()), 4))

    # Figure F13
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=300)
    seasons = ("All", "Summer", "Monsoon", "Winter")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for season, color in zip(seasons, colors):
        ax.plot(prof_df.minutes_before_target, prof_df[season], label=season, color=color, lw=1.2)
    ax.set_xlabel("Minutes before the target")
    ax.set_ylabel("Mean attention weight")
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.axvspan(60, 0, color="gray", alpha=0.12, linestyle="--", linewidth=0.8)

    ax_inset = ax.inset_axes([0.42, 0.38, 0.52, 0.52])
    for season, color in zip(seasons, colors):
        ax_inset.plot(prof_df.minutes_before_target, prof_df[season], color=color, lw=1.2)
    ax_inset.set_xlim(60, 0)
    ax_inset.set_title("0-60 min detail", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, linestyle=":", alpha=0.5)

    f13_path = os.path.join(D_FIG, "F13_attention_profile_inset.png")
    fig.savefig(f13_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {f13_path}")
    print("s07 complete.")


if __name__ == "__main__":
    main()
