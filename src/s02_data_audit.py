"""
s02_data_audit.py
=================
Step 2 -- Data and information-set audit.

Purpose:
  - Audits the APSLDC dataset for timestamp continuity, flat runs,
    and feature timing patterns
  - Documents the exact chronological partition boundaries
  - Computes feature manifest and scaler-scope comparison
  - Reads receptive field directly from the DTAQ checkpoint configs

Reviewer comments addressed: R1-1, R1-2, R1-8, R1-10, R2-1

Output (tables/):
  T01_partition_manifest.csv|docx
  T02_feature_manifest.csv|docx
  T_scaler_scope.csv|docx
  T_weather_timing.csv|docx
  T_receptive_field.csv|docx
  results/data_audit.json
  results/flat_runs.csv
  results/weather_alignment_head.csv

Runtime: ~2 min on CPU.

Usage:
    python s02_data_audit.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Set P3_DATA environment variable. See README.md."
        )

    df, feats, ap = load_frame()
    idx = df.index
    v   = df["Actual_MW"].values
    audit = {}

    # ---------------------------------------------------------------- timestamps
    grid = pd.date_range(idx.min(), idx.max(), freq="15min")
    steps = pd.Series(idx).diff().dropna()
    audit["rows"]                        = int(len(df))
    audit["first"]                       = str(idx.min())
    audit["last"]                        = str(idx.max())
    audit["expected_rows_on_15min_grid"] = int(len(grid))
    audit["missing_timestamps"]          = int(len(grid.difference(idx)))
    audit["duplicate_timestamps"]        = int(idx.duplicated().sum())
    audit["non_15min_steps"]             = int((steps != pd.Timedelta("15min")).sum())
    audit["sorted_unique_index"]         = bool(idx.is_monotonic_increasing)
    miss = grid.difference(idx)
    audit["missing_examples"]            = [str(t) for t in miss[:10]]

    # ---------------------------------------------------------------- flat runs (identical consecutive values)
    same = np.r_[False, np.diff(v) == 0]
    runs = []
    i = 0
    while i < len(v):
        if same[i]:
            j = i
            while j < len(v) and same[j]:
                j += 1
            runs.append((i - 1, j - i + 1))
            i = j
        else:
            i += 1

    flat = pd.DataFrame(
        [(str(idx[s]), L, v[s]) for s, L in runs if L >= 8],
        columns=["start", "length_steps", "value_MW"]
    ).sort_values("length_steps", ascending=False)

    audit["flat_runs_ge8"]           = int(len(flat))
    audit["flat_steps_total"]        = int(flat["length_steps"].sum()) if len(flat) else 0
    audit["longest_flat_run_steps"]  = int(flat["length_steps"].max()) if len(flat) else 0

    med = pd.Series(v).rolling(672, center=True, min_periods=96).median().values
    audit["points_below_50pct_of_local_median"] = int(np.sum(v < 0.5 * med))
    audit["min_MW"] = float(v.min())
    audit["max_MW"] = float(v.max())
    flat.head(20).to_csv(os.path.join(D_RES, "flat_runs.csv"), index=False)

    # ---------------------------------------------------------------- partition manifest
    def manifest(lb, n_rows):
        n = n_rows - lb
        a, b = legacy_bounds(n)
        ts   = idx.values[lb:]
        mk   = lambda name, s, e: dict(
            partition=name,
            first_window=int(s), last_window=int(e - 1), n_windows=int(e - s),
            first_target=str(ts[s]), last_target=str(ts[e - 1]),
            pct_of_windows=round(100 * (e - s) / n, 2)
        )
        return [mk("train", 0, a), mk("validation/calibration", a, b), mk("test", b, n)]

    part_rows = []
    for lb in (1, 48):
        for r in manifest(lb, len(df)):
            part_rows.append(dict(protocol="legacy", look_back=lb, **r))

    dev_s, cal_s, test_s = strict_bounds(df, feats)
    ts48 = idx.values[48:]
    sb = {
        "train":       (0,                              int(np.searchsorted(ts48, dev_s))),
        "dev":         (int(np.searchsorted(ts48, dev_s)), int(np.searchsorted(ts48, cal_s))),
        "calibration": (int(np.searchsorted(ts48, cal_s)), int(np.searchsorted(ts48, test_s))),
        "test":        (int(np.searchsorted(ts48, test_s)), len(ts48)),
    }
    for k, (s, e) in sb.items():
        part_rows.append(dict(
            protocol="strict", look_back=48, partition=k,
            first_window=s, last_window=e - 1, n_windows=e - s,
            first_target=str(ts48[s]), last_target=str(ts48[e - 1]),
            pct_of_windows=round(100 * (e - s) / len(ts48), 2)
        ))

    man = pd.DataFrame(part_rows)
    write_table(
        "T01_partition_manifest", man,
        "Exact chronological partitions (windows are assigned by target timestamp).",
        "Legacy: test = int(0.2 n), validation = int(0.1 (n - test)); scaler fitted on all rows. "
        "Strict: scaler fitted on rows before the dev block; "
        "input windows may reach back into the preceding partition (past values only)."
    )

    # ---------------------------------------------------------------- feature manifest
    grp  = feature_groups(feats)
    rule = {
        "hour":           "hour of day of the 15-min stamp (0-23)",
        "dow":            "day of week (0=Monday) from the calendar",
        "is_weekend":     "1 if Saturday/Sunday",
        "precip_binary":  "0 if preciptype is none/empty else 1",
        "conditions_enc": "ordinal 0-5: Clear, Partially cloudy, Overcast, Rain+Partially cloudy, Rain+Overcast, Rain",
        "icon_enc":       "ordinal 0-4: clear, partly-cloudy, cloudy, fog/wind, rain",
        "dow_sin":        "sin(2*pi*dow/7)",
        "dow_cos":        "cos(2*pi*dow/7)",
        "is_holiday":     "1 if holiday_status is a holiday",
    }
    fm = []
    for j, f in enumerate(feats):
        cls = next(k for k, lst in grp.items() if f in lst)
        fm.append(dict(
            index=j, name=f, group=cls,
            derived=f in rule and f in ("precip_binary", "conditions_enc", "icon_enc", "dow_sin", "dow_cos", "is_holiday"),
            rule=rule.get(f, "as provided (numeric column of the cleaned file)"),
            in_C3_top10=f in TOP10_LEGACY,
            std_over_all_rows=float(df[f].std()),
            mean_over_all_rows=float(df[f].mean())
        ))
    write_table(
        "T02_feature_manifest", pd.DataFrame(fm), "The 26 input features.",
        "All features enter standardised; window inputs use rows t-LB ... t-1 only."
    )

    # ---------------------------------------------------------------- scaler scope
    from sklearn.preprocessing import StandardScaler
    a48, _  = legacy_bounds(len(df) - 48)
    tr_rows = idx.values < ts48[a48]
    full    = StandardScaler().fit(df[feats].values)
    trn     = StandardScaler().fit(df[feats].values[tr_rows])
    sh = pd.DataFrame(dict(
        feature=feats,
        mean_shift_in_train_sd=(full.mean_ - trn.mean_) / trn.scale_,
        std_ratio_full_over_train=full.scale_ / trn.scale_
    ))
    write_table(
        "T_scaler_scope", sh,
        "Effect of fitting the scaler on all rows (legacy) versus training rows only."
    )
    audit["scaler_max_abs_mean_shift_sd"] = float(sh["mean_shift_in_train_sd"].abs().max())

    # ---------------------------------------------------------------- weather timing
    wc = grp["weather_numeric"] + grp["weather_encoded"]
    al = []
    for c in wc:
        ch    = df[c].diff().fillna(0).ne(0)
        n_ch  = int(ch.sum())
        at0   = float((idx[ch.values].minute == 0).mean()) if n_ch else float("nan")
        al.append(dict(
            column=c, n_changes=n_ch,
            share_of_changes_at_minute_0=at0,
            pattern=("hourly step (changes only on the hour)" if at0 >= 0.95
                     else "changes inside the hour")
        ))
    alt = pd.DataFrame(al)
    write_table("T_weather_timing", alt, "Timing pattern of weather columns relative to the 15-min grid.")
    k = int(np.argmax((idx.minute == 0) & (np.arange(len(idx)) > 8)))
    df.iloc[k - 4:k + 8][["Actual_MW"] + wc[:3]].to_csv(
        os.path.join(D_RES, "weather_alignment_head.csv")
    )
    audit["weather_columns_hourly_step"] = int((alt["share_of_changes_at_minute_0"] >= 0.95).sum())
    audit["weather_columns_total"]       = len(wc)

    # ---------------------------------------------------------------- receptive field from checkpoints
    rf = []
    for cond in CONDS:
        pth = ckpt_path(cond, "DTAQ")
        if os.path.exists(pth):
            rows_, r = conv_audit(pth)
            k_sizes  = [x[1][0] if isinstance(x[1], (list, tuple)) else x[1] for x in rows_]
            dils     = [x[2][0] if isinstance(x[2], (list, tuple)) else x[2] for x in rows_]
            rf.append(dict(
                checkpoint=f"{cond}_DTAQ",
                n_conv_layers=len(rows_),
                n_causal=sum(1 for x in rows_ if x[3] == "causal"),
                dilations=str([d for i, d in enumerate(dils) if rows_[i][3] == "causal"]),
                kernel_sizes=str(sorted(set(k_sizes))),
                receptive_field=r
            ))
    if rf:
        write_table(
            "T_receptive_field", pd.DataFrame(rf),
            "Receptive field computed from the checkpoint configuration (1 + sum (k-1) d)."
        )

    with open(os.path.join(D_RES, "data_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    print("\nData audit complete.")
    for k, v in audit.items():
        if isinstance(v, (int, float, bool, str)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
