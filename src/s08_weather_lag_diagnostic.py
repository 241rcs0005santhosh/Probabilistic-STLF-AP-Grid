"""
s08_weather_lag_diagnostic.py
=============================
Step 8 -- Inference-time weather perturbation and information-set diagnostics (C2).

Purpose:
  Evaluates reliance of the trained C2 models (LSTM-Q, CNN-BiGRU-Q, DTAQ) on
  unlagged weather inputs by shifting weather features at inference time.
  Variants:
    - baseline (recorded inputs)
    - all weather lagged 1 h (4 steps)
    - all weather lagged 3 h (12 steps)
    - precipitation lagged 1 h
    - all weather set to feature mean (scaled 0.0)

Output:
  tables/T_information_set_diagnostic.csv|docx

"""

import os
import sys
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_tf import *


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nSet P3_DATA environment variable.")

    df, feats, ap = load_frame()
    p = prepare(df, feats, 48, "legacy")
    te = p.idx["test"]
    grp = feature_groups(feats)
    W_COLS = grp["weather_numeric"] + grp["weather_encoded"]
    P_COLS = [c for c in ("precip", "precipprob", "precip_binary") if c in feats]
    colidx = {f: i for i, f in enumerate(feats)}
    base_vals = p.scaler.transform(df[feats].values.astype(np.float64)).astype(np.float32)

    def shift_cols(vals, cols, k):
        out = vals.copy()
        ix = [colidx[c] for c in cols]
        out[k:, ix] = vals[:-k, ix]
        out[:k, ix] = vals[0, ix]
        return out

    def zero_cols(vals, cols):
        out = vals.copy()
        out[:, [colidx[c] for c in cols]] = 0.0
        return out

    BASE = "baseline (recorded inputs)"
    VARIANTS = {
        BASE: base_vals,
        "all weather lagged 1 h": shift_cols(base_vals, W_COLS, 4),
        "all weather lagged 3 h": shift_cols(base_vals, W_COLS, 12),
        "precipitation lagged 1 h": shift_cols(base_vals, P_COLS, 4),
        "all weather set to mean": zero_cols(base_vals, W_COLS)
    }

    y_te = inv(p, p.y[te])
    rmse = lambda yy, pp: float(np.sqrt(np.mean((yy - pp) ** 2)))
    B = 200 if QUICK else 2000
    rows, checks = [], []

    for a in ARCHS:
        z = load_preds(f"legacy_C2_{a}")
        qh = cqr_qhat(cqr_scores(z["y_cal"], z["yq_cal"]))
        m = load_ckpt(ckpt_path("C2", a))
        med_base = None

        for vname, vals in VARIANTS.items():
            Xall, _, _ = make_windows(vals, 48)
            yq = q_to_mw(p, predict_q(m, Xall[te]))
            if vname == BASE:
                med_base = yq[:, MED].copy()
                base_rmse = rmse(y_te, med_base)
                checks.append(dict(model=a, max_abs_diff_vs_cache_MW=float(np.max(np.abs(yq - z["yq_test"])))))
                d, lo, hi = 0.0, 0.0, 0.0
            else:
                d = rmse(y_te, yq[:, MED]) - base_rmse
                lo, hi = block_boot_ci(
                    lambda yy, b_, c_: rmse(yy, c_) - rmse(yy, b_),
                    [y_te, med_base, yq[:, MED]], block=96, B=B
                )
            st = shift_outer(yq, qh)
            rows.append(dict(
                model=a, variant=vname,
                RMSE=rmse(y_te, yq[:, MED]), MAE=point_metrics(y_te, yq[:, MED])["MAE"],
                dRMSE_vs_recorded=d, dRMSE_CI_lo=lo, dRMSE_CI_hi=hi,
                PICP_fixed_calibration=picp(y_te, st[:, LO], st[:, HI]),
                Winkler90=winkler(y_te, st[:, LO], st[:, HI])
            ))
        del m
        keras.backend.clear_session()
        gc.collect()

    print(pd.DataFrame(checks).to_string(index=False))
    assert all(c["max_abs_diff_vs_cache_MW"] < 1e-2 for c in checks), "baseline does not match cache"

    write_table(
        "T_information_set_diagnostic", pd.DataFrame(rows),
        "Inference-time weather perturbations for the legacy C2 models (diagnostic).",
        "Models were trained on the recorded inputs. Changes in RMSE show reliance on current weather values, not "
        "the accuracy of a lagged-information forecaster. Calibration is kept at the recorded values."
    )
    print("s08 complete.")


if __name__ == "__main__":
    main()
