"""
s05_frozen_sensitivity.py
=========================
Step 5 -- Sensitivity to repeated (frozen) actual load values.

Purpose:
  Repeated identical consecutive 15-min values may indicate sensor issues.
  This script evaluates whether model results change materially when such
  targets are excluded from the test set (and from calibration).

  Rule: a "run" = consecutive identical 15-min values.
  Flagged = repeated values of runs with at least MIN_RUN identical values.
  The first value of each run is kept (not flagged).

Output:
  tables/T_frozen_sensitivity.csv|docx
  tables/T_frozen_DM.csv|docx
  results/frozen_runs_min8.csv

"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

THRESHOLDS = [4, 8, 16]
PRIMARY    = 8   # threshold used in the main tables


def frozen_flags(values, min_run):
    """
    Flag repeated values in runs of at least min_run identical consecutive values.
    The first value of each run is kept (not flagged).
    """
    v    = np.asarray(values, float)
    rep  = np.r_[False, np.diff(v) == 0]
    flags = np.zeros(len(v), bool)
    i = 0
    while i < len(v):
        if rep[i]:
            j = i
            while j < len(v) and rep[j]:
                j += 1
            if (j - i + 1) >= min_run:
                flags[i:j] = True
            i = j
        else:
            i += 1
    return flags


def runs_table(flags, index):
    rows, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            rows.append((index[i], index[j - 1], j - i))
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=["first_flagged", "last_flagged", "flagged_steps"])


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Set P3_DATA environment variable. See README.md."
        )

    df, feats, ap = load_frame()
    FLAG = {
        m: pd.Series(frozen_flags(df["Actual_MW"].values, m), index=df.index)
        for m in THRESHOLDS
    }
    runs_table(FLAG[PRIMARY].values, df.index).to_csv(
        os.path.join(D_RES, f"frozen_runs_min{PRIMARY}.csv"), index=False
    )
    print("flagged steps on the full series:", {m: int(FLAG[m].sum()) for m in THRESHOLDS})

    store, rows = {}, []
    for cond in ("C2", "C3"):
        for arch in ARCHS:
            tag = f"legacy_{cond}_{arch}"
            try:
                z = load_preds(tag)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Cache not found: {tag}\nRun s01_legacy_inference.py first."
                )
            store[(cond, arch)] = z

            ts    = pd.DatetimeIndex(z["ts_test"])
            tsc   = pd.DatetimeIndex(z["ts_cal"])
            y     = z["y_test"].astype(float)
            yq    = z["yq_test"].astype(float)
            yc    = z["y_cal"].astype(float)
            yqc   = z["yq_cal"].astype(float)
            qh_all = cqr_qhat(cqr_scores(yc, yqc))
            st    = shift_outer(yq, qh_all)

            rows.append(dict(
                condition=cond, model=arch,
                subset="all targets", min_run="-", calibration="all windows",
                n=len(y), flagged_share_test=0.0,
                **point_metrics(y, yq[:, MED]),
                PICP_static=picp(y, st[:, LO], st[:, HI]),
                Winkler90=winkler(y, st[:, LO], st[:, HI]),
                q_hat=qh_all
            ))

            for m in THRESHOLDS:
                fts = FLAG[m].reindex(ts)
                fcs = FLAG[m].reindex(tsc)
                assert fts.notna().all() and fcs.notna().all(), \
                    "timestamp not found in the full series"
                ft, fc = fts.values.astype(bool), fcs.values.astype(bool)
                keep   = ~ft
                qh_u   = cqr_qhat(cqr_scores(yc[~fc], yqc[~fc]))

                for calname, qh in (("all windows", qh_all), ("unflagged windows only", qh_u)):
                    s2 = shift_outer(yq[keep], qh)
                    rows.append(dict(
                        condition=cond, model=arch,
                        subset=f"unflagged (min_run={m})", min_run=m,
                        calibration=calname, n=int(keep.sum()),
                        flagged_share_test=float(ft.mean()),
                        **point_metrics(y[keep], yq[keep, MED]),
                        PICP_static=picp(y[keep], s2[:, LO], s2[:, HI]),
                        Winkler90=winkler(y[keep], s2[:, LO], s2[:, HI]),
                        q_hat=qh
                    ))

    T = pd.DataFrame(rows)
    write_table(
        "T_frozen_sensitivity", T,
        "Model results on all test targets and on targets outside flagged frozen runs.",
        "Flagged = repeated values of runs with at least min_run identical 15-min values, "
        "taken from the full recorded series. Evaluation excludes flagged targets; "
        "the 'unflagged windows only' rows also drop flagged windows from calibration. "
        "No model was retrained."
    )

    # Diebold-Mariano on the same targets for each subset
    dm_rows = []
    for cond in ("C2", "C3"):
        dq  = store[(cond, "DTAQ")]
        ts  = pd.DatetimeIndex(dq["ts_test"])
        y   = dq["y_test"].astype(float)
        pdq = dq["yq_test"][:, MED].astype(float)
        subsets = [(
            "all targets", np.ones(len(y), bool)
        )] + [(
            f"unflagged (min_run={m})",
            ~FLAG[m].reindex(ts).values.astype(bool)
        ) for m in THRESHOLDS]

        for other in ("LSTM_Q", "CNN_BiGRU_Q"):
            po = store[(cond, other)]["yq_test"][:, MED].astype(float)
            for label, keep in subsets:
                r = dm_hac(y[keep], pdq[keep], po[keep])
                dm_rows.append(dict(
                    condition=cond, comparison=f"DTAQ vs {other}",
                    subset=label, n=r["n"],
                    lag=r["lag"], DM_HAC=r["DM"], p=r["p"],
                    mean_loss_diff=r["mean_diff"],
                    CI_lo=r["CI_lo"], CI_hi=r["CI_hi"]
                ))

    write_table(
        "T_frozen_DM", pd.DataFrame(dm_rows),
        "Diebold-Mariano (HAC) tests on all targets and on unflagged targets.",
        "Negative DM = DTAQ has smaller squared errors. "
        "p-values below 0.0001 are shown as 0.0000."
    )
    print(f"s05 done. Check: the 'all targets' RMSE should match T03 (DTAQ C2 about 173.88).")


if __name__ == "__main__":
    main()
