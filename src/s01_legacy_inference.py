"""
s01_legacy_inference.py
"""

import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_tf import *   # imports common.py helpers + TF helpers


# ------------------------------------------------------------------ reference values
# These are the manuscript-reported values (Tables 4 and 8) used ONLY for pass/fail comparison.
# They are NOT used in computing results -- they serve only to verify reproduction.
REF = {
    # (RMSE, MAE, MAPE, R2, PICP_raw, PICP_cqr, q_hat)
    "C1_LSTM_Q":       (217.28, 112.20, 1.2843, 0.9746, 0.8961, 0.8925,  -1.80),
    "C1_CNN_BiGRU_Q":  (216.69, 108.67, 1.2425, 0.9747, 0.9481, 0.9155, -47.12),
    "C1_DTAQ":         (214.58, 109.20, 1.2416, 0.9752, 0.9032, 0.9165,  14.84),
    "C2_LSTM_Q":       (206.54, 104.51, 1.2079, 0.9771, 0.9056, 0.8851, -11.57),
    "C2_CNN_BiGRU_Q":  (199.99, 105.17, 1.2087, 0.9785, 0.9378, 0.9126, -23.05),
    "C2_DTAQ":         (173.88,  77.16, 0.9035, 0.9837, 0.8791, 0.8926,   6.96),
    "C3_LSTM_Q":       (207.11,  98.13, 1.1301, 0.9769, 0.9138, 0.8951, -12.61),
    "C3_CNN_BiGRU_Q":  (189.34,  90.48, 1.0390, 0.9807, 0.9424, 0.9234, -21.05),
    "C3_DTAQ":         (173.82,  75.55, 0.8860, 0.9837, 0.8382, 0.8824,  17.75),
}

REF_PARAMS = {
    "LSTM_Q":      {"C1": 25607,  "C2": 25607,  "C3": 21511},
    "CNN_BiGRU_Q": {"C1": 72199,  "C2": 72199,  "C3": 69127},
    "DTAQ":        {"C1": 127207, "C2": 127207,  "C3": 123111},
}

# Relaxed tolerances to allow minor float deviations from legacy system executions
TOL = dict(RMSE=0.5, MAE=1.0, MAPE=0.02, R2=0.005, PICP=0.01, QH=2.0)


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Set the P3_DATA environment variable to the path of APSLDC_15min_Clean.xlsx.\n"
            "See README.md -> Dataset Preparation for instructions."
        )

    print("Loading dataset...")
    df, feats, ap = load_frame()
    print(f"features: {feats}")

    rows = []
    for cond in CONDS:
        feat_set = TOP10_LEGACY if cond == "C3" else feats
        p = prepare(df, feat_set, LB_OF[cond], "legacy")
        print(
            f"\n{cond}: windows={len(p.y):,}  "
            f"train={len(p.idx['train']):,}  "
            f"cal={len(p.idx['cal']):,}  "
            f"test={len(p.idx['test']):,}"
        )

        for arch in ARCHS:
            tag  = f"{cond}_{arch}"
            path = ckpt_path(cond, arch)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {path}\n"
                    "Ensure all 9 .keras files are in the checkpoints/ directory."
                )

            m = load_ckpt(path)
            y_t, yq_t, qh = export_preds(
                m, p, f"legacy_{tag}",
                extra=dict(params=np.array([m.count_params()]))
            )
            r = summarize(tag, y_t, yq_t, qh, dict(params=int(m.count_params()), n_test=len(y_t)))

            ref = REF[tag]
            chk = dict(
                RMSE=abs(r["RMSE"] - ref[0]) <= TOL["RMSE"],
                MAE=abs(r["MAE"]  - ref[1]) <= TOL["MAE"],
                MAPE=abs(r["MAPE"] - ref[2]) <= TOL["MAPE"],
                R2=abs(r["R2"]   - ref[3]) <= TOL["R2"],
                PICP_raw=abs(r["PICP_raw"] - ref[4]) <= TOL["PICP"],
                PICP_cqr=abs(r["PICP_cqr"] - ref[5]) <= TOL["PICP"],
                q_hat=abs(r["q_hat"]       - ref[6]) <= TOL["QH"],
                params=r["params"] == REF_PARAMS[arch][cond],
            )
            r.update({f"ok_{k}": v for k, v in chk.items()})
            r["ALL_PASS"] = all(chk.values())
            rows.append(r)

            status = "PASS" if r["ALL_PASS"] else "FAIL " + str([k for k, v in chk.items() if not v])
            print(
                f"  {tag:16s}  RMSE={r['RMSE']:.2f}  "
                f"PICP raw/cqr={r['PICP_raw']:.4f}/{r['PICP_cqr']:.4f}  "
                f"q_hat={qh:+.2f}  params={r['params']:,}  -> {status}"
            )
            del m
            keras.backend.clear_session()
            gc.collect()

        del p
        gc.collect()

    import pandas as pd
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(D_RES, "m00_reproduction_check.csv"), index=False)
    write_table(
        "T03_reproduction_check",
        res[["tag", "RMSE", "MAE", "MAPE", "R2", "PICP_raw", "PICP_cqr", "q_hat", "params", "ALL_PASS"]],
        "Reproduction of the legacy results from the original checkpoints."
    )

    if not res["ALL_PASS"].all():
        failing = res.loc[~res["ALL_PASS"], "tag"].tolist()
        raise AssertionError(
            f"REPRODUCTION FAILED for: {failing}\n"
            "Do not run later scripts until this is resolved."
        )

    print("\nALL 9 CHECKPOINTS REPRODUCE THE MANUSCRIPT VALUES")
    return res


if __name__ == "__main__":
    main()
