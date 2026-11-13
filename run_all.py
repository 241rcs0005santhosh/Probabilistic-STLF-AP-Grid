"""
run_all.py
==========
Master reproduction orchestrator for:
"DTAQ-based Probabilistic Short-Term Load Forecasting"

Executes all analysis and reproduction modules in logical execution sequence:
  Step 1: s01_legacy_inference.py       -- Verify checkpoints & reproduce manuscript results
  Step 2: s02_data_audit.py             -- Timestamp audit, partitions, receptive field
  Step 3: s03_cqr_diagnostics.py        -- Conformal diagnostics & prospective calibration
  Step 4: s04_hac_dm_baselines.py       -- HAC Diebold-Mariano tests & naive baselines
  Step 5: s05_frozen_sensitivity.py     -- Repeated-value / flat-run sensitivity
  Step 6: s06_permutation_importance.py  -- Permutation feature ranking
  Step 7: s07_attention_perturbation.py -- Bahdanau attention perturbation & profiles
  Step 8: s08_weather_lag_diagnostic.py -- Inference-time weather information-set audit
  Step 9: s09_component_ablation.py     -- Sequential component ablation (TCN, NCQ, Attn, CQR)

Usage:
    python run_all.py
    python run_all.py --quick     # Fast smoke test
"""

import os
import sys
import argparse
import subprocess
import time


STEPS = [
    ("s01_legacy_inference.py",       "Reproduction check & raw inference from 9 checkpoints"),
    ("s02_data_audit.py",             "Dataset, timestamp, partition, and receptive field audit"),
    ("s03_cqr_diagnostics.py",        "CQR diagnostics, prospective calibration (Rolling/ACI/Mondrian)"),
    ("s04_hac_dm_baselines.py",       "HAC Diebold-Mariano tests, naive baselines, ramps & peaks"),
    ("s05_frozen_sensitivity.py",     "Repeated load value (flat runs) sensitivity evaluation"),
    ("s06_permutation_importance.py", "Permutation feature importance and provenance analysis"),
    ("s07_attention_perturbation.py", "At-inference attention weight perturbation diagnostics"),
    ("s08_weather_lag_diagnostic.py", "Weather latency and information-set diagnostics"),
    ("s09_component_ablation.py",     "Component ablation: TCN-Point, TCN-Q, DTAQ-noCQR, DTAQ-CQR"),
]


def main():
    parser = argparse.ArgumentParser(description="Run complete DTAQ reproduction pipeline.")
    parser.add_argument("--quick", action="store_true", help="Run in quick smoke-test mode.")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.quick:
        env["P3_QUICK"] = "1"

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir  = os.path.join(repo_dir, "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    print("=" * 75)
    print("DTAQ: PROBABILISTIC SHORT-TERM LOAD FORECASTING REPRODUCTION PIPELINE")
    print(f"Mode: {'QUICK SMOKE-TEST' if args.quick else 'FULL PUBLICATION REPRODUCTION'}")
    print(f"Repository Root: {repo_dir}")
    print("=" * 75)

    total_start = time.time()
    for idx, (script, desc) in enumerate(STEPS, start=1):
        script_path = os.path.join(src_dir, script)
        print(f"\n[{idx}/{len(STEPS)}] Running {script} -- {desc}")
        print("-" * 75)
        t0 = time.time()
        ret = subprocess.run([sys.executable, script_path], env=env)
        elapsed = time.time() - t0
        if ret.returncode != 0:
            print(f"\n[ERROR] Step {script} failed with exit code {ret.returncode} after {elapsed:.1f}s.")
            sys.exit(ret.returncode)
        print(f"[{idx}/{len(STEPS)}] {script} completed successfully in {elapsed:.1f}s.")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"ALL REPRODUCTION STEPS COMPLETED SUCCESSFULLY in {total_elapsed / 60:.1f} min.")
    print("Results written to:")
    print("  - Tables:  results/tables/")
    print("  - Figures: results/figures/")
    print("  - Data:    results/results/")
    print("  - Cache:   results/cache/")
    print("=" * 75)


if __name__ == "__main__":
    main()
