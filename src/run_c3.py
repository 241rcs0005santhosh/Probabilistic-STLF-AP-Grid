"""
run_c3.py
=========
Experiment C3: Look-back = 48 steps (12 hours), Top-10 Features.

Evaluates LSTM-Q, CNN-BiGRU-Q, and proposed DTAQ under the C3 condition:
  - Input shape: (48, 10)
  - Features: Top-10 features selected by permutation importance
      ['Actual_MW', 'solarenergy', 'solarradiation', 'hour', 'TEMP',
       'dew', 'humidity', 'feelslike', 'uvindex', 'is_weekend']
  - Split: 72% Train / 08% Cal / 20% Test (chronological)
  - Models evaluated: LSTM-Q, CNN-BiGRU-Q, DTAQ
  - Applies Conformalized Quantile Regression (CQR)
  - Evaluates both raw and calibrated prediction intervals

Usage:
    python src/run_c3.py
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


def run_experiment_c3():
    print("=" * 70)
    print("EXPERIMENT C3: Look-back = 48 steps (12 h), Top-10 Features")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nSet P3_DATA environment variable.")

    df, feats, ap = load_frame()
    p = prepare(df, TOP10_LEGACY, lb=48, protocol="legacy")
    print(f"C3 sequence count: {len(p.y):,} windows (train={len(p.idx['train']):,}, "
          f"cal={len(p.idx['cal']):,}, test={len(p.idx['test']):,})")
    print(f"Features: {TOP10_LEGACY}")

    results = []
    for arch in ARCHS:
        tag  = f"C3_{arch}"
        path = ckpt_path("C3", arch)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        m = load_ckpt(path)
        y_t, yq_t, qh = export_preds(
            m, p, f"legacy_{tag}",
            extra=dict(params=np.array([m.count_params()]))
        )
        r = summarize(tag, y_t, yq_t, qh, dict(condition="C3", model=arch, params=int(m.count_params())))
        results.append(r)
        print(f"  {arch:<15} RMSE={r['RMSE']:>7.2f} MW  MAPE={r['MAPE']:>6.4f}%  "
              f"PICP_raw={r['PICP_raw']:.4f}  PICP_cqr={r['PICP_cqr']:.4f}  q_hat={qh:>+6.2f} MW")
        del m
        keras.backend.clear_session()
        gc.collect()

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(D_RES, "experiment_C3_results.csv")
    df_res.to_csv(out_csv, index=False)
    write_table("T_experiment_C3", df_res[["model", "RMSE", "MAE", "MAPE", "R2", "PICP_raw", "PICP_cqr", "q_hat"]],
                "Experiment C3 Results (Look-back = 48 steps, Top-10 features)")
    print(f"Results saved to: {out_csv}")
    return df_res


if __name__ == "__main__":
    run_experiment_c3()
