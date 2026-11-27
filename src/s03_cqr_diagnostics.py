"""
s03_cqr_diagnostics.py
======================
Step 3 -- Conformal diagnostics + prospective calibration.

Purpose:
  - Signed-score CQR diagnostics (q_hat sign, shortfall, block-bootstrap CI)
  - Prospective methods: rolling-window split-conformal and ACI
  - Mondrian CQR (season x peak conditional calibration)
  - Conditional coverage tables (by season, hour, demand decile, peak/off-peak)
  - Pinball reconciliation against manuscript Table 5 values

Output (tables/, figures/):
  T04_cqr_diagnostics.csv|docx
  T05_calibration_methods.csv|docx
  T05_sensitivity_grid.csv
  T06_conditional_coverage.csv|docx
  T07_pinball_reconciliation.csv|docx
  T05c_label_delay.csv|docx (C2_DTAQ only)
  F02_reliability_C2.png, F02b_reliability_C3.png
  F03_monthly_coverage_C2_DTAQ.png
  F04_rolling_picp_C2_DTAQ.png
  F05_coverage_by_decile_C2_DTAQ.png
  F05b_coverage_by_season_peak_C2_DTAQ.png
  F06_winkler_by_method_C2_DTAQ.png
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

B = 200 if QUICK else 2000

# W and gamma grid for prospective method selection (on the last 30% of calibration block)
W_GRID  = [288, 576, 1440, 2880, 5760]
GAMMAS  = [0.001, 0.005, 0.01, 0.02, 0.05]
DELAY   = 1   # label delay: score at t is usable only from t+1 (labels revealed on-the-fly)

# Manuscript Table 5 average pinball values (for reconciliation only)
MAN_T6_PINBALL = {
    "C1_LSTM_Q": 35.154, "C1_CNN_BiGRU_Q": 35.667, "C1_DTAQ": 33.742,
    "C2_LSTM_Q": 32.890, "C2_CNN_BiGRU_Q": 33.040, "C2_DTAQ": 25.171,
    "C3_LSTM_Q": 31.770, "C3_CNN_BiGRU_Q": 29.490, "C3_DTAQ": 24.009,
}


def yq_with(yq, lo, hi):
    """Return a copy of yq with the outer quantiles replaced."""
    out = yq.copy()
    out[:, LO] = lo
    out[:, HI] = hi
    return out


def select_params(yc, yqc):
    """
    Select W (rolling window) and (gamma, W) for ACI on the last 30% of the
    calibration block only. Returns (best_W_rolling, (best_gamma, best_W_ACI)).
    """
    n      = len(yc)
    sel_n  = max(int(n * 0.3), 1)
    yc_s   = yc[-sel_n:]
    yqc_s  = yqc[-sel_n:]
    sc_s   = cqr_scores(yc_s, yqc_s)
    sc_all = cqr_scores(yc, yqc)
    # Warm-start: use all earlier calibration scores as the initial buffer
    init   = sc_all[:-sel_n].tolist()

    best_W_r  = None
    best_wink_r = np.inf
    for W in W_GRID:
        lo, hi, _, _ = prospective_conformal(yc_s, yqc_s, init, W=W, delay=DELAY)
        wk = winkler(yc_s, lo, hi)
        if wk < best_wink_r:
            best_wink_r = wk
            best_W_r    = W

    best_g    = None
    best_W_a  = None
    best_wink_a = np.inf
    for W in W_GRID:
        for g in GAMMAS:
            lo, hi, _, _ = prospective_conformal(yc_s, yqc_s, init, W=W, gamma=g, delay=DELAY)
            wk = winkler(yc_s, lo, hi)
            if wk < best_wink_a:
                best_wink_a = wk
                best_g      = g
                best_W_a    = W

    return best_W_r, (best_g, best_W_a)


def grp_labels(ts, y):
    """Group labels for conditional coverage."""
    ts_pd = pd.DatetimeIndex(ts)
    return dict(
        month=ts_pd.month,
        season=season_of(ts),
        hour_block=(ts_pd.hour // 3 * 3).astype(str) + "h",
        demand_decile=pd.qcut(y, 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop").astype(str),
        weekday=np.where(ts_pd.weekday < 5, "Weekday", "Weekend"),
        peak=np.where(is_peak(ts), "Peak (18-21h)", "Off-peak"),
    )


def main():
    # Load cached predictions from s01
    try:
        D = {(c, a): load_preds(tagof(c, a)) for c in CONDS for a in ARCHS}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cache file not found: {e}\n"
            "Run s01_legacy_inference.py first."
        )

    t04, t05, t05g, t06, t07, store = [], [], [], [], [], {}

    for (c, a), z in D.items():
        name = f"{c}_{a}"
        yc, yqc = z["y_cal"], z["yq_cal"]
        y,  yq  = z["y_test"], z["yq_test"]
        ts       = z["ts_test"]

        sc  = cqr_scores(yc, yqc)
        qh  = cqr_qhat(sc)
        assert abs(qh - float(z["q_hat"][0])) < 1e-6, "q_hat mismatch with cache"

        stat     = shift_outer(yq, qh)
        cov_stat = covered(y, stat[:, LO], stat[:, HI])
        cov_raw  = covered(y, yq[:, LO], yq[:, HI])
        ci       = block_boot_ci(lambda x: x.mean(), [cov_stat.astype(float)], block=96, B=B)

        t04.append(dict(
            model=name, q_hat_MW=qh,
            effect="widens interval (raw under-covers)" if qh > 0 else "narrows interval (raw over-covers)",
            PICP_calibration_block_raw=picp(yc, yqc[:, LO], yqc[:, HI]),
            PICP_test_raw=cov_raw.mean(), PICP_test_cqr=cov_stat.mean(),
            shortfall_vs_0p90=0.90 - cov_stat.mean(),
            CI95_lo=ci[0], CI95_hi=ci[1], n_test=len(y)
        ))

        # Pinball reconciliation
        pr, pc = pinball_avg(y, yq), pinball_avg(y, stat)
        dr, dc = abs(MAN_T6_PINBALL[name] - pr), abs(MAN_T6_PINBALL[name] - pc)
        tol    = 0.05
        label  = (
            "matches RAW"  if dr <= tol and dc > tol else
            "matches CQR"  if dc <= tol and dr > tol else
            "matches BOTH" if dr <= tol else
            "matches NEITHER (manuscript value unexplained)"
        )
        t07.append(dict(
            model=name, pinball7_raw=pr, pinball7_cqr=pc, raw_minus_cqr=pr - pc,
            manuscript_table6_value=MAN_T6_PINBALL[name],
            abs_diff_vs_raw=dr, abs_diff_vs_cqr=dc, tolerance=tol,
            manuscript_value_status=label
        ))

        # Prospective methods
        (W_r, (g_a, W_a)) = select_params(yc, yqc)
        methods = {"static CQR": (stat, "q_hat fixed from the whole calibration block")}

        lo, hi, _, _ = prospective_conformal(y, yq, sc, W=W_r, delay=DELAY)
        methods["rolling CQR"] = (yq_with(yq, lo, hi), f"W={W_r}, delay={DELAY}")

        lo, hi, qh_t, al_t = prospective_conformal(y, yq, sc, W=W_a, gamma=g_a, delay=DELAY)
        methods["ACI"] = (yq_with(yq, lo, hi), f"gamma={g_a}, W={W_a}, delay={DELAY}")

        gts = grp_labels(ts, y)
        gtc = grp_labels(z["ts_cal"], yc)
        gc_ = gtc["season"].astype(object) + "|" + gtc["peak"]
        gt_ = gts["season"].astype(object) + "|" + gts["peak"]
        qh_m, _ = mondrian_qhat(sc, gc_, gt_)
        methods["Mondrian CQR (season x peak)"] = (
            yq_with(yq, yq[:, LO] - qh_m, yq[:, HI] + qh_m),
            "q_hat per season x peak cell (calibration block)"
        )

        for mname, (yqa, par) in methods.items():
            cv = covered(y, yqa[:, LO], yqa[:, HI])
            ci = block_boot_ci(lambda x: x.mean(), [cv.astype(float)], block=96, B=B)
            t05.append(dict(
                model=name, method=mname, parameters=par,
                PICP=cv.mean(), CI95_lo=ci[0], CI95_hi=ci[1],
                **{k: v for k, v in prob_metrics(y, yqa).items() if k != "PICP"}
            ))
            store[(name, mname)] = (cv, yqa)
            for gt, labs in gts.items():
                if mname in ("static CQR", "ACI"):
                    for g in np.unique(labs):
                        m2 = labs == g
                        t06.append(dict(
                            model=name, method=mname, group_type=gt, group=g,
                            n=int(m2.sum()), PICP=cv[m2].mean(),
                            PINAW=float(np.mean(yqa[m2, HI] - yqa[m2, LO]) / (y.max() - y.min()))
                        ))

        # Full sensitivity grid
        for W in W_GRID:
            lo, hi, _, _ = prospective_conformal(y, yq, sc, W=W, delay=DELAY)
            t05g.append(dict(model=name, method="rolling", gamma="", W=W,
                             selected=(W == W_r), PICP=picp(y, lo, hi), Winkler90=winkler(y, lo, hi)))
            for g in GAMMAS:
                lo, hi, _, _ = prospective_conformal(y, yq, sc, W=W, gamma=g, delay=DELAY)
                t05g.append(dict(model=name, method="ACI", gamma=g, W=W,
                                 selected=(W == W_a and g == g_a),
                                 PICP=picp(y, lo, hi), Winkler90=winkler(y, lo, hi)))

        print(
            f"  {name:16s}  q_hat={qh:+7.2f}  raw={picp(y, yq[:, LO], yq[:, HI]):.4f}  "
            f"static/rolling/ACI/Mondrian=" +
            " / ".join(f"{store[(name, m2)][0].mean():.4f}" for m2 in methods)
        )

        # Label-delay sensitivity (C2_DTAQ only)
        if name == "C2_DTAQ":
            dly = []
            for d in (1, 4, 96):
                lo, hi, _, _ = prospective_conformal(y, yq, sc, W=W_a, gamma=g_a, delay=d)
                dly.append(dict(label_delay_steps=d, method="ACI",
                                PICP=picp(y, lo, hi), Winkler90=winkler(y, lo, hi)))
                lo, hi, _, _ = prospective_conformal(y, yq, sc, W=W_r, delay=d)
                dly.append(dict(label_delay_steps=d, method="rolling",
                                PICP=picp(y, lo, hi), Winkler90=winkler(y, lo, hi)))
            write_table(
                f"T05c_label_delay{SUF}", pd.DataFrame(dly),
                "Sensitivity of prospective calibration to the label delay (DTAQ, C2)."
            )

    # Save tables
    write_table(f"T04_cqr_diagnostics{SUF}", pd.DataFrame(t04),
                "Signed-score CQR diagnostics.",
                "q_hat > 0 widens and q_hat < 0 narrows the interval. CI: moving-block bootstrap (block=96).")
    write_table(f"T05_calibration_methods{SUF}", pd.DataFrame(t05),
                "Calibration methods compared on the test block.",
                "W and gamma were selected on the last 30% of the calibration block only.")
    pd.DataFrame(t05g).to_csv(os.path.join(D_RES, f"T05_sensitivity_grid{SUF}.csv"), index=False)
    write_table(f"T06_conditional_coverage{SUF}", pd.DataFrame(t06),
                "Coverage by month, season, hour block, demand decile, weekday/weekend and peak.")
    write_table(f"T07_pinball_reconciliation{SUF}", pd.DataFrame(t07),
                "Average 7-quantile pinball loss on raw and CQR-adjusted quantiles.")

    # Figures
    cols = {"LSTM_Q": "tab:blue", "CNN_BiGRU_Q": "tab:green", "DTAQ": "tab:red"}
    for cond in ("C2", "C3"):
        fig, ax = new_fig(3.5, 3.0)
        for a in ARCHS:
            z  = D[(cond, a)]
            st = shift_outer(z["yq_test"], cqr_qhat(cqr_scores(z["y_cal"], z["yq_cal"])))
            ax.plot(QUANTILES, [np.mean(z["y_test"] <= st[:, i]) for i in range(Q)],
                    "o-", ms=3, color=cols[a], label=a.replace("_", "-"))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("Nominal quantile level"); ax.set_ylabel("Empirical coverage"); ax.legend(frameon=False)
        save_fig(fig, f"F02{'b' if cond == 'C3' else ''}_reliability_{cond}{SUF}",
                 "T05/T04", f"Reliability of the quantile forecasts after CQR ({cond}).")

    print("s03 done.")


if __name__ == "__main__":
    main()
