"""
s06_permutation_importance.py
=============================
Step 6 -- Permutation feature importance for DTAQ C2.

Purpose:
  Computes permutation feature importance on training windows (and on test
  windows as a diagnostic) for the DTAQ C2 model (48 look-back steps, 26 features).
  Assesses ranking stability across repeated shuffles (Spearman rho, Kendall tau)
  and evaluates feature provenance against manuscript Table 5.

Output:
  tables/T14_permutation_importance.csv|docx
  results/s06_feature_provenance.json

"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_tf import *

N_SAMPLE   = 2000
K_REPEATS  = 10

# Manuscript Table 3 reported values (from Manuscript_Revision_Pass1 Table 3)
MANUSCRIPT_TABLE3 = {
    "Actual_MW": 1799.71, "hour": 20.29, "solarenergy": 8.90, "solarradiation": 8.28,
    "uvindex": 5.44, "windspeed": 3.90, "feelslike": 2.82, "TEMP": 2.55,
    "sealevelpressure": 2.01, "icon_enc": 1.90
}

# Legacy Submission 2 values (for historical provenance audit)
MANUSCRIPT_T5_LEGACY = {
    "Actual_MW": 1718.16, "solarenergy": 114.63, "solarradiation": 88.98, "hour": 42.53, "TEMP": 40.12,
    "dew": 33.83, "humidity": 32.75, "feelslike": 15.79, "uvindex": 7.94, "is_weekend": 4.81, "sealevelpressure": 3.80,
    "severerisk": 3.12, "conditions_enc": 3.09, "windspeed": 2.93, "cloudcover": 2.66, "precip": 2.54, "precipprob": 1.70,
    "dow": 1.59, "winddir": 0.73, "visibility": 0.47, "icon_enc": 0.22, "dow_sin": 0.07, "is_holiday": 0.00,
    "precip_binary": 0.00, "dow_cos": -0.32, "windgust": -0.51
}


def stability(imp):
    K = len(imp)
    ranks = np.argsort(np.argsort(-imp, axis=1), axis=1) + 1
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    rho = float(np.mean([spearmanr(imp[a], imp[b]).correlation for a, b in pairs]))
    tau = float(np.mean([kendalltau(ranks[a], ranks[b]).correlation for a, b in pairs]))
    return rho, tau


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}\nSet P3_DATA environment variable.")
    ckpt_file = ckpt_path("C2", "DTAQ")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print("Loading data and model...")
    df, feats, ap = load_frame()
    p = prepare(df, feats, 48, "legacy")
    model = load_ckpt(ckpt_file)

    grp = feature_groups(feats)
    group_of = {f: next(k for k, v in grp.items() if f in v) for f in feats}
    jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))

    rng = np.random.default_rng(SEED)

    def draw(part):
        ii = p.idx[part]
        return np.sort(rng.choice(ii, size=min(N_SAMPLE, len(ii)), replace=False))

    res = {}
    for part in ("train", "test"):
        sel = draw(part)
        reseed(SEED)
        base, imp = perm_importance(model, p.X[sel], inv(p, p.y[sel]), p, K=K_REPEATS, seed=SEED)
        res[part] = dict(base=base, imp=imp, n=len(sel), stab=stability(imp))
        print(f"  {part}: n={len(sel)}  baseline RMSE={base:.2f}  "
              f"Spearman={res[part]['stab'][0]:.3f}  Kendall={res[part]['stab'][1]:.3f}")

    tr, te = res["train"], res["test"]
    mtr, sdtr = tr["imp"].mean(0), tr["imp"].std(0, ddof=1)
    mte, sdte = te["imp"].mean(0), te["imp"].std(0, ddof=1)
    order_tr, order_te = np.argsort(-mtr), np.argsort(-mte)
    rk_tr = np.empty(len(feats), int)
    rk_tr[order_tr] = np.arange(1, len(feats) + 1)
    rk_te = np.empty(len(feats), int)
    rk_te[order_te] = np.arange(1, len(feats) + 1)
    top_tr = [feats[i] for i in order_tr[:10]]
    top_te = [feats[i] for i in order_te[:10]]

    print("\nTop-10 on training windows:", top_tr)
    print("Top-10 on test windows    :", top_te)
    print("Jaccard(train top-10, checkpoint C3 list)    =", round(jac(top_tr, TOP10_LEGACY), 3))
    print("Jaccard(train top-10, manuscript Table 3 list)=", round(jac(top_tr, MANUSCRIPT_TABLE3_FEATURES), 3))
    print("Jaccard(train top-10, test top-10)           =", round(jac(top_tr, top_te), 3))

    T = pd.DataFrame([dict(
        feature=f, group=group_of[f],
        dRMSE_train_mean=mtr[i], dRMSE_train_sd=sdtr[i], rank_train=rk_tr[i],
        dRMSE_test_mean=mte[i], dRMSE_test_sd=sdte[i], rank_test=rk_te[i],
        in_c3_checkpoint_list=f in TOP10_LEGACY,
        in_manuscript_table3=f in MANUSCRIPT_TABLE3_FEATURES,
        manuscript_table3_dRMSE=MANUSCRIPT_TABLE3.get(f, np.nan),
        legacy_submission2_dRMSE=MANUSCRIPT_T5_LEGACY.get(f, np.nan)
    ) for i, f in enumerate(feats)]).sort_values("rank_train")

    write_table(
        "T14_permutation_importance", T,
        "Permutation importance for DTAQ (C2), computed on training windows.",
        f"Mean and SD of the RMSE increase over {K_REPEATS} repeats, {N_SAMPLE} windows per partition. "
        f"Rank stability across repeats (training): Spearman {res['train']['stab'][0]:.3f}, Kendall {res['train']['stab'][1]:.3f}. "
        f"Top-10 overlap with C3 checkpoints: {jac(top_tr, TOP10_LEGACY):.2f}; with Table 3: {jac(top_tr, MANUSCRIPT_TABLE3_FEATURES):.2f}."
    )

    with open(os.path.join(D_RES, "s06_feature_provenance.json"), "w") as f:
        json.dump(dict(
            top10_train=top_tr, top10_test=top_te,
            c3_checkpoint_features=TOP10_LEGACY,
            manuscript_table3_features=MANUSCRIPT_TABLE3_FEATURES,
            jaccard_train_checkpoint=jac(top_tr, TOP10_LEGACY),
            jaccard_train_table3=jac(top_tr, MANUSCRIPT_TABLE3_FEATURES),
            jaccard_test_checkpoint=jac(top_te, TOP10_LEGACY)
        ), f, indent=1)

    print("s06 complete.")


if __name__ == "__main__":
    main()
