"""
run_c1.py
=========
Experiment C1: Look-back = 1 step (15-min lag), All 26 Features.

Evaluates LSTM-Q, CNN-BiGRU-Q, and DTAQ under the C1 condition:
  - Input shape: (1, 26)
  - Split: 70% Train / 10% Cal / 20% Test (chronological)
  - Models evaluated: LSTM-Q, CNN-BiGRU-Q, DTAQ
  - Applies Conformalized Quantile Regression (CQR)
  - Evaluates both raw and calibrated prediction intervals

Usage:
    python src/run_c1.py
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from common_tf import *


def run_experiment_c1():
    print("=" * 70)
    print("EXPERIMENT C1: Look-back = 1 step (15 min), All 26 Features")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nSet P3_DATA environment variable.")

    df, feats, ap = load_frame()
    p = prepare(df, feats, lb=1, protocol="legacy")
    print(f"C1 sequence count: {len(p.y):,} windows (train={len(p.idx['train']):,}, "
          f"cal={len(p.idx['cal']):,}, test={len(p.idx['test']):,})")

    results = []
    for arch in ARCHS:
        tag  = f"C1_{arch}"
        path = ckpt_path("C1", arch)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        m = load_ckpt(path)
        y_t, yq_t, qh = export_preds(
            m, p, f"legacy_{tag}",
            extra=dict(params=np.array([m.count_params()]))
        )
        r = summarize(tag, y_t, yq_t, qh, dict(condition="C1", model=arch, params=int(m.count_params())))
        results.append(r)
        print(f"  {arch:<15} RMSE={r['RMSE']:>7.2f} MW  MAPE={r['MAPE']:>6.4f}%  "
              f"PICP_raw={r['PICP_raw']:.4f}  PICP_cqr={r['PICP_cqr']:.4f}  q_hat={qh:>+6.2f} MW")
        del m
        keras.backend.clear_session()
        gc.collect()

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(D_RES, "experiment_C1_results.csv")
    df_res.to_csv(out_csv, index=False)
    write_table("T_experiment_C1", df_res[["model", "RMSE", "MAE", "MAPE", "R2", "PICP_raw", "PICP_cqr", "q_hat"]],
                "Experiment C1 Results (Look-back = 1 step, 26 features)")
    print(f"Results saved to: {out_csv}")
    return df_res


if __name__ == "__main__":
    run_experiment_c1()
