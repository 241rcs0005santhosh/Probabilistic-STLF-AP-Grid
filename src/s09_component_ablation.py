"""
s09_component_ablation.py
=========================
Step 9 -- Component ablation study (TCN, Non-crossing Quantile Head, Attention, CQR).

Purpose:
  Sequentially isolates the contribution of each architectural module of DTAQ:
    1. TCN-Point: Dilated TCN backbone + Dense(1) -> point forecast only
    2. TCN-Q: Dilated TCN backbone + NonCrossingQuantileHead -> 7 quantiles (no attention)
    3. DTAQ-noCQR: Full DTAQ (TCN + Bahdanau temporal attention + NonCrossingQuantileHead) without CQR
    4. DTAQ [C2]: Full DTAQ + Post-hoc Conformalized Quantile Regression (CQR)

Reviewer comments addressed: R1-4, R2-2

Prerequisites:
  - APSLDC_15min_Clean.xlsx available
  - checkpoints/C2_DTAQ.keras available

Output:
  tables/T11_component_study.csv|docx
  results/Paper3_Ablation_Results.xlsx

Usage:
    python s09_component_ablation.py
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_tf import *

TCN_FILTERS   = 64
TCN_KERNEL    = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]
PROJ_UNITS    = 32
DROP_RATE     = 0.1


def tcn_residual_block(x, filters, kernel_size, dilation, pfx):
    residual = x
    y = layers.Conv1D(filters, kernel_size, dilation_rate=dilation,
                      padding="causal", activation="relu", name=f"{pfx}_conv")(x)
    y = layers.BatchNormalization(name=f"{pfx}_bn")(y)
    y = layers.Dropout(DROP_RATE, name=f"{pfx}_drop")(y)
    if x.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding="same", name=f"{pfx}_skip")(x)
    return layers.Add(name=f"{pfx}_add")([y, residual])


def build_tcn_backbone(inp, pfx="tcn"):
    x = layers.Conv1D(TCN_FILTERS, 1, padding="same", name=f"{pfx}_proj")(inp)
    for i, d in enumerate(TCN_DILATIONS):
        x = tcn_residual_block(x, TCN_FILTERS, TCN_KERNEL, d, pfx=f"{pfx}_b{i+1}_d{d}")
    return x


def build_tcn_point(lb, nf, steps_per_epoch):
    inp  = layers.Input(shape=(lb, nf), name="input")
    tcn  = build_tcn_backbone(inp, pfx="tcnp")
    gap  = layers.GlobalAveragePooling1D(name="gap")(tcn)
    proj = layers.Dense(PROJ_UNITS, activation="relu", name="proj")(gap)
    out  = layers.Dense(1, name="point_out")(proj)
    model = keras.Model(inp, out, name="TCN_Point")
    lr   = keras.optimizers.schedules.CosineDecayRestarts(1e-3, first_decay_steps=10 * steps_per_epoch, alpha=1e-5)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model


def build_tcn_q(lb, nf, steps_per_epoch):
    inp  = layers.Input(shape=(lb, nf), name="input")
    tcn  = build_tcn_backbone(inp, pfx="tcnq")
    gap  = layers.GlobalAveragePooling1D(name="gap")(tcn)
    proj = layers.Dense(PROJ_UNITS, activation="relu", name="proj")(gap)
    drop = layers.Dropout(DROP_RATE, name="drop")(proj)
    out  = NonCrossingQuantileHead(QUANTILES, name="quantile_head")(drop)
    model = keras.Model(inp, out, name="TCN_Q")
    lr   = keras.optimizers.schedules.CosineDecayRestarts(1e-3, first_decay_steps=10 * steps_per_epoch, alpha=1e-5)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=pinball_avg_loss(QUANTILES))
    return model


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nSet P3_DATA environment variable.")
    ckpt_file = ckpt_path("C2", "DTAQ")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print("Loading dataset for ablation study...")
    df, feats, ap = load_frame()
    p = prepare(df, feats, 48, "legacy")
    tr, ca, te = p.idx["train"], p.idx["cal"], p.idx["test"]

    Xtr, ytr = p.X[tr], p.y[tr]
    Xvl, yvl = p.X[ca], p.y[ca]
    Xte, yte = p.X[te], p.y[te]

    yt_test = inv(p, yte)
    yt_val  = inv(p, yvl)

    results = {}
    preds   = {}
    batch_size = 256
    steps_per_epoch = max(1, len(Xtr) // batch_size)
    epochs = 2 if QUICK else 60

    # 1. TCN-Point
    print("\n--- Evaluating TCN-Point ---")
    reseed(SEED); keras.backend.clear_session(); gc.collect()
    m_p = build_tcn_point(48, len(feats), steps_per_epoch)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)]
    t0 = time.time()
    h = m_p.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)
    tt = round(time.time() - t0, 1)
    y_pt_mw = inv(p, m_p.predict(Xte, batch_size=512, verbose=0).ravel())
    pt_p = point_metrics(yt_test, y_pt_mw)
    results["TCN-Point"] = dict(
        **pt_p,
        PICP="N/A", PI_MW="N/A", PINAW="N/A", Pinball="N/A",
        PICP_CQR="N/A", PINAW_CQR="N/A", q_hat_MW="N/A",
        Params=m_p.count_params(), Epochs=len(h.history["loss"]), Train_s=tt,
        Note="Deterministic -- no prediction interval"
    )
    print(f"  TCN-Point: RMSE={pt_p['RMSE']:.2f} MW, MAPE={pt_p['MAPE']:.4f}%")
    del m_p; keras.backend.clear_session(); gc.collect()

    # 2. TCN-Q
    print("\n--- Evaluating TCN-Q ---")
    reseed(SEED); keras.backend.clear_session(); gc.collect()
    m_q = build_tcn_q(48, len(feats), steps_per_epoch)
    t0 = time.time()
    h = m_q.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)
    tt = round(time.time() - t0, 1)
    yq_te_q = q_to_mw(p, predict_q(m_q, Xte))
    yq_vl_q = q_to_mw(p, predict_q(m_q, Xvl))
    pt_q = point_metrics(yt_test, yq_te_q[:, MED])
    pr_q = prob_metrics(yt_test, yq_te_q)
    qh_q = cqr_qhat(cqr_scores(yt_val, yq_vl_q))
    yq_cal_q = shift_outer(yq_te_q, qh_q)
    pr_cal_q = prob_metrics(yt_test, yq_cal_q)
    results["TCN-Q"] = dict(
        **pt_q, **pr_q,
        PICP_CQR=pr_cal_q["PICP"], PINAW_CQR=pr_cal_q["PINAW"],
        q_hat_MW=round(qh_q, 4),
        Params=m_q.count_params(), Epochs=len(h.history["loss"]), Train_s=tt,
        Note="TCN + NCQ head, no attention"
    )
    print(f"  TCN-Q: RMSE={pt_q['RMSE']:.2f} MW, PICP_raw={pr_q['PICP']:.4f}, PICP_CQR={pr_cal_q['PICP']:.4f}")
    del m_q; keras.backend.clear_session(); gc.collect()

    # 3 & 4. DTAQ from pretrained checkpoint
    print("\n--- Evaluating DTAQ (checkpoint) ---")
    reseed(SEED); keras.backend.clear_session(); gc.collect()
    m_dtaq = load_ckpt(ckpt_file)
    yq_te_dtaq = q_to_mw(p, predict_q(m_dtaq, Xte))
    yq_vl_dtaq = q_to_mw(p, predict_q(m_dtaq, Xvl))
    pt_dtaq = point_metrics(yt_test, yq_te_dtaq[:, MED])
    pr_dtaq_raw = prob_metrics(yt_test, yq_te_dtaq)
    qh_dtaq = cqr_qhat(cqr_scores(yt_val, yq_vl_dtaq))
    yq_cal_dtaq = shift_outer(yq_te_dtaq, qh_dtaq)
    pr_dtaq_cal = prob_metrics(yt_test, yq_cal_dtaq)

    results["DTAQ-noCQR"] = dict(
        **pt_dtaq, **pr_dtaq_raw,
        PICP_CQR="N/A", PINAW_CQR="N/A", q_hat_MW="N/A",
        Params=m_dtaq.count_params(), Epochs="pre-trained", Train_s="pre-trained",
        Note="Full DTAQ -- no conformal calibration"
    )
    results["DTAQ [C2]"] = dict(
        **pt_dtaq, **pr_dtaq_raw,
        PICP_CQR=pr_dtaq_cal["PICP"], PINAW_CQR=pr_dtaq_cal["PINAW"],
        q_hat_MW=round(qh_dtaq, 4),
        Params=m_dtaq.count_params(), Epochs="pre-trained", Train_s="pre-trained",
        Note="Full DTAQ + conformal calibration (CQR)"
    )
    print(f"  DTAQ-noCQR: RMSE={pt_dtaq['RMSE']:.2f} MW, PICP_raw={pr_dtaq_raw['PICP']:.4f}")
    print(f"  DTAQ [C2]:  RMSE={pt_dtaq['RMSE']:.2f} MW, PICP_CQR={pr_dtaq_cal['PICP']:.4f}, q_hat={qh_dtaq:+.2f} MW")

    del m_dtaq; keras.backend.clear_session(); gc.collect()

    # Build summary table
    ABLATION_ORDER = ["TCN-Point", "TCN-Q", "DTAQ-noCQR", "DTAQ [C2]"]
    COMP_FLAGS = {
        "TCN-Point":  ("Yes", "No",  "No",  "No"),
        "TCN-Q":      ("Yes", "Yes", "No",  "No"),
        "DTAQ-noCQR": ("Yes", "Yes", "Yes", "No"),
        "DTAQ [C2]":  ("Yes", "Yes", "Yes", "Yes"),
    }
    rows = []
    for name in ABLATION_ORDER:
        r = results[name]
        tcn, ncq, att, cqr = COMP_FLAGS[name]
        rows.append({
            "Model": name, "TCN": tcn, "NCQ_Head": ncq, "Attention": att, "CQR": cqr,
            "Params": r["Params"], "RMSE_MW": r["RMSE"], "MAE_MW": r["MAE"],
            "MAPE_pct": r["MAPE"], "R2": r["R2"],
            "PICP_raw": r.get("PICP", "N/A"), "PINAW": r.get("PINAW", "N/A"),
            "Pinball": r.get("Pinball7", r.get("Pinball", "N/A")),
            "PICP_CQR": r.get("PICP_CQR", "N/A"), "q_hat_MW": r.get("q_hat_MW", "N/A"),
            "Note": r.get("Note", "")
        })
    abl_df = pd.DataFrame(rows)
    write_table(
        "T11_component_study", abl_df,
        "Component ablation study of DTAQ [C2]: isolating TCN, NCQ head, attention, and CQR."
    )
    print("\n" + abl_df.to_string(index=False))
    print("s09 complete.")


if __name__ == "__main__":
    main()
