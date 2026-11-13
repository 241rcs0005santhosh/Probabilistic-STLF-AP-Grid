"""
s04_hac_dm_baselines.py
=======================
Step 4 -- HAC Diebold-Mariano tests, naive/operator baselines, ramps & peaks.

Purpose:
  - HAC-corrected Diebold-Mariano tests (Newey-West Bartlett bandwidth)
  - Holm family-wise error correction over 10 pre-declared comparisons
  - Naive baselines: persistence, seasonal-naive 24h/7d
  - Probabilistic naive baseline: persistence + empirical residual quantiles
  - Performance on extreme ramp and peak subsets

Reviewer comments addressed: R1-5, R1-9, R2-5, R2-6

Prerequisites: s01_legacy_inference.py must be run first.

Output:
  tables/T08_hac_dm.csv|docx
  tables/T09_baselines.csv|docx
  tables/T09b_probabilistic_baselines.csv|docx
  tables/T10_ramps_peaks.csv|docx
  results/loss_differential_series.csv
  figures/F07_dm_forest_C2.png
  figures/F08_error_acf_C2_DTAQ.png
  figures/F09_baselines_rmse.png
  figures/F10_ramp_peak_coverage_C2.png

Runtime: ~5 min on CPU.

Usage:
    python s04_hac_dm_baselines.py
    P3_PROTOCOL=strict python s04_hac_dm_baselines.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

PROTOCOL = os.environ.get("P3_PROTOCOL", "legacy")
SUF      = "" if PROTOCOL == "legacy" else "_strict"
tagof    = (lambda c, a: f"legacy_{c}_{a}") if PROTOCOL == "legacy" \
           else (lambda c, a: f"strict_{c}_{a}_s{SEEDS[0]}")
B        = 200 if QUICK else 2000


def main():
    # Load cached predictions
    try:
        D = {(c, a): load_preds(tagof(c, a)) for c in ("C2", "C3") for a in ARCHS}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cache file not found: {e}\nRun s01_legacy_inference.py first.")

    z0  = D[("C2", "DTAQ")]
    y   = z0["y_test"]
    ts  = pd.DatetimeIndex(z0["ts_test"])
    for k, z in D.items():
        assert np.allclose(z["y_test"], y), "C2/C3 test targets differ"

    # Naive baselines (timestamp-based look-ups)
    ts_all  = np.concatenate([z0["ts_cal"], z0["ts_test"]])
    y_all   = np.concatenate([z0["y_cal"], y])
    persist = lagged_lookup(ts_all, y_all, ts, pd.Timedelta("15min"))
    naive1d = lagged_lookup(ts_all, y_all, ts, pd.Timedelta("1D"))
    naive7d = lagged_lookup(ts_all, y_all, ts, pd.Timedelta("7D"))

    common_mask = ~(np.isnan(persist) | np.isnan(naive1d) | np.isnan(naive7d))
    n_c         = int(common_mask.sum())
    print(f"targets: {len(y):,};  with all naive look-ups available: {n_c:,}")

    preds = {f"{c}_{a}": D[(c, a)]["yq_test"][:, MED] for (c, a) in D}
    preds.update({"persistence": persist, "seasonal-naive 24 h": naive1d, "seasonal-naive 7 d": naive7d})

    # ---------------------------------------------------------------- T09 baselines
    rows = []
    for k, p in preds.items():
        m = common_mask
        rows.append(dict(forecaster=k, n=n_c, **point_metrics(y[m], p[m])))

    # Try loading operator forecast
    try:
        df, feats, ap = load_frame()
        if ap is not None:
            op = df[ap].reindex(pd.DatetimeIndex(ts)).values
            mo = common_mask & ~np.isnan(op)
            rows.append(dict(
                forecaster="APSLDC operator forecast "
                           "(issue time / horizon: TO BE CONFIRMED BY AUTHORS - information set NOT matched)",
                n=int(mo.sum()),
                **point_metrics(y[mo], op[mo])
            ))
    except Exception as e:
        print(f"  (raw data not available - operator forecast skipped: {str(e)[:60]})")

    # Probabilistic naive: persistence + empirical residual quantiles
    pc   = lagged_lookup(ts_all, y_all, pd.DatetimeIndex(z0["ts_cal"]), pd.Timedelta("15min"))
    ok   = ~np.isnan(pc)
    res  = z0["y_cal"][ok] - pc[ok]
    qlo, qhi = np.quantile(res, [0.05, 0.95])
    lo_p, hi_p = persist + qlo, persist + qhi
    m = common_mask
    prob = [dict(
        forecaster="persistence + empirical residual quantiles",
        PICP=picp(y[m], lo_p[m], hi_p[m]),
        PINAW=float(np.mean(hi_p[m] - lo_p[m]) / (y.max() - y.min())),
        Winkler90=winkler(y[m], lo_p[m], hi_p[m])
    )]
    for (c, a), z in D.items():
        st = shift_outer(z["yq_test"], cqr_qhat(cqr_scores(z["y_cal"], z["yq_cal"])))
        prob.append(dict(
            forecaster=f"{c}_{a} (static CQR)",
            PICP=picp(y[m], st[m, LO], st[m, HI]),
            PINAW=float(np.mean(st[m, HI] - st[m, LO]) / (y.max() - y.min())),
            Winkler90=winkler(y[m], st[m, LO], st[m, HI])
        ))

    write_table(f"T09_baselines{SUF}", pd.DataFrame(rows),
                "Point accuracy of naive baselines and models on the identical test targets.")
    write_table(f"T09b_probabilistic_baselines{SUF}", pd.DataFrame(prob),
                "Interval metrics: empirical-residual persistence versus calibrated models.")

    # ---------------------------------------------------------------- T08 HAC DM + Holm
    fam = []
    for c in ("C2", "C3"):
        for other in ["LSTM_Q", "CNN_BiGRU_Q", "persistence", "seasonal-naive 24 h", "seasonal-naive 7 d"]:
            p2 = preds[f"{c}_{other}"] if other in ARCHS else preds[other]
            fam.append((c, other, preds[f"{c}_DTAQ"][m], p2[m]))

    out, series = [], {}
    for c, other, p1, p2 in fam:
        yy   = y[m]
        r    = dm_hac(yy, p1, p2)
        lags = {L: dm_hac(yy, p1, p2, lag=L)["DM"] for L in (48, 96, 192)}
        ae   = dm_hac(yy, p1, p2, loss="ae")
        rm1, rm2 = np.sqrt(np.mean((yy - p1) ** 2)), np.sqrt(np.mean((yy - p2) ** 2))
        ci = block_boot_ci(
            lambda a_, b_, c_: 100 * (1 - np.sqrt(np.mean((a_ - b_) ** 2)) / np.sqrt(np.mean((a_ - c_) ** 2))),
            [yy, p1, p2], block=96, B=B
        )
        out.append(dict(
            condition=c, comparison=f"DTAQ vs {other}", n=r["n"],
            RMSE_DTAQ=rm1, RMSE_other=rm2,
            improvement_pct=100 * (1 - rm1 / rm2),
            improvement_CI95_lo=ci[0], improvement_CI95_hi=ci[1],
            HAC_lag_auto=r["lag"], DM_HAC=r["DM"], p_HAC=r["p"],
            mean_loss_diff_MW2=r["mean_diff"], diff_CI95_lo=r["CI_lo"], diff_CI95_hi=r["CI_hi"],
            DM_lag48=lags[48], DM_lag96=lags[96], DM_lag192=lags[192],
            DM_abs_error=ae["DM"], DM_iid=dm_iid(yy, p1, p2)[0]
        ))
        if other in ARCHS:
            series[f"{c}_vs_{other}"] = (yy - p1) ** 2 - (yy - p2) ** 2

    T08 = pd.DataFrame(out)
    T08["p_Holm"] = holm(T08["p_HAC"].values)
    T08["significant_all_lags_0p05"] = [
        all(abs(T08.loc[i, k]) > 1.96 for k in ("DM_HAC", "DM_lag48", "DM_lag96", "DM_lag192"))
        for i in range(len(T08))
    ]
    write_table(f"T08_hac_dm{SUF}", T08,
                "Diebold-Mariano tests with Newey-West HAC variance.",
                "d = squared-error(DTAQ) - squared-error(other); negative = DTAQ better. "
                "Holm correction over the 10 pre-declared comparisons.")
    pd.DataFrame({"timestamp": ts[m], **{k: v for k, v in series.items()}}).to_csv(
        os.path.join(D_RES, f"loss_differential_series{SUF}.csv"), index=False
    )

    # ---------------------------------------------------------------- ramps and peaks
    r1   = np.abs(y - persist)
    h1   = lagged_lookup(ts_all, y_all, ts, pd.Timedelta("1h"))
    r4   = np.abs(y - h1)
    subsets = {
        "all targets":          np.ones(len(y), bool),
        "top 1% one-step ramps": r1 >= np.nanpercentile(r1, 99),
        "top 5% one-step ramps": r1 >= np.nanpercentile(r1, 95),
        "top 5% one-hour ramps": r4 >= np.nanpercentile(r4, 95),
        "top load decile":       y  >= np.percentile(y, 90),
        "peak hours 18-21 h":   is_peak(ts),
    }
    rr = []
    for (c, a), z in D.items():
        st = shift_outer(z["yq_test"], cqr_qhat(cqr_scores(z["y_cal"], z["yq_cal"])))
        for sname, msk in subsets.items():
            msk2 = msk & ~np.isnan(r1) if "ramp" in sname else msk
            yy   = y[msk2]
            cv   = covered(yy, st[msk2, LO], st[msk2, HI]).astype(float)
            ci   = block_boot_ci(lambda x: x.mean(), [cv], block=24, B=min(B, 1000)) \
                   if len(cv) >= 100 else (float("nan"), float("nan"))
            rr.append(dict(
                condition=c, model=a, subset=sname, n=int(msk2.sum()),
                **{k: v for k, v in point_metrics(yy, z["yq_test"][msk2, MED]).items() if k in ("RMSE", "MAE")},
                PICP=cv.mean(), PICP_CI95_lo=ci[0], PICP_CI95_hi=ci[1],
                Winkler90=winkler(yy, st[msk2, LO], st[msk2, HI])
            ))

    write_table(f"T10_ramps_peaks{SUF}", pd.DataFrame(rr),
                "Accuracy and coverage on extreme-ramp and peak subsets (static CQR).",
                "CI for PICP: block bootstrap (block=24) on the subset order; "
                "NaN where the subset has fewer than 100 windows.")

    # Figures
    fig, ax = new_fig(3.5, 3.4)
    t8  = T08[T08.condition == "C2"]
    yy_ = np.arange(len(t8))[::-1]
    ax.errorbar(t8.improvement_pct, yy_,
                xerr=[t8.improvement_pct - t8.improvement_CI95_lo,
                      t8.improvement_CI95_hi - t8.improvement_pct],
                fmt="o", color="tab:red", capsize=2)
    ax.set_yticks(yy_)
    ax.set_yticklabels([s.replace("DTAQ vs ", "vs ") for s in t8.comparison])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("RMSE improvement of DTAQ (%)")
    save_fig(fig, f"F07_dm_forest_C2{SUF}", "T08",
             "RMSE improvement of DTAQ (C2) with moving-block-bootstrap 95% CI.")

    fig, ax = new_fig(3.5, 2.7)
    e  = y[m] - preds["C2_DTAQ"][m]
    a_ = acf(e, 192)
    ax.vlines(range(193), 0, a_, lw=0.8)
    ax.axhline(1.96 / np.sqrt(len(e)), color="r", ls="--", lw=0.8)
    ax.axhline(-1.96 / np.sqrt(len(e)), color="r", ls="--", lw=0.8)
    ax.set_xlabel("Lag (15-min steps)"); ax.set_ylabel("ACF of forecast error")
    save_fig(fig, f"F08_error_acf_C2_DTAQ{SUF}", "loss_differential_series.csv",
             "Autocorrelation of the DTAQ (C2) forecast error; dashed = iid 95% bounds.")

    t9  = pd.DataFrame(rows)
    fig, ax = new_fig(3.5, 3.0)
    sel = t9[~t9.forecaster.str.contains("operator") & ~t9.forecaster.str.startswith("C3")]
    ax.barh(range(len(sel)), sel.RMSE, color="tab:gray")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([s.replace("C2_", "").replace("_", "-") for s in sel.forecaster])
    ax.set_xlabel("RMSE (MW)"); ax.invert_yaxis()
    save_fig(fig, f"F09_baselines_rmse{SUF}", "T09", "RMSE of naive baselines and C2 models on identical targets.")

    t10 = pd.DataFrame(rr)
    t10 = t10[t10.condition == "C2"]
    fig, ax = new_fig(7, 2.8)
    w = 0.27
    for i, a in enumerate(ARCHS):
        s_ = t10[t10.model == a]
        ax.bar(np.arange(len(s_)) + (i - 1) * w, s_.PICP, w, label=a.replace("_", "-"))
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(s_)))
    ax.set_xticklabels(s_.subset, rotation=20, ha="right")
    ax.set_ylabel("PICP (static CQR)"); ax.legend(frameon=False, ncol=3)
    save_fig(fig, f"F10_ramp_peak_coverage_C2{SUF}", "T10", "Coverage on extreme-ramp and peak subsets (C2).")

    print("s04 done.")


if __name__ == "__main__":
    main()
