# "Probabilistic Short-Term Load Forecasting with Dilated Temporal Convolutional Networks and Conformal Calibration: Evidence from the Andhra Pradesh State Grid"

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation and reproduction repository for the research paper:  
**"Probabilistic Short-Term Load Forecasting with Dilated Temporal Convolutional Networks and Conformal Calibration: Evidence from the Andhra Pradesh State Grid"**  
Submitted to *International Journal of Intelligent Engineering and Systems* (IJIES).

---

## 1. Overview

This repository provides the complete, self-contained implementation of the **DTAQ** architecture and its comparative baselines (**LSTM-Q** and **CNN-BiGRU-Q**) for 15-minute resolution Short-Term Load Forecasting (STLF), evaluated on the Andhra Pradesh State Load Despatch Centre (APSLDC) electrical grid.

### Key Innovations & Architecture
* **Dilated Temporal Convolutional Network (TCN):** 5-block causal residual TCN with exponential dilations $d \in \{1, 2, 4, 8, 16\}$, kernel size $k=3$, achieving a theoretical receptive field of 125 steps (capped at 48 historical steps = 12 hours).
* **Bahdanau Temporal Attention:** Context-aware temporal pooling over TCN hidden states to focus on critical load transitions.
* **Non-crossing Quantile Head (`NonCrossingQuantileHead`):** Strict structural guarantee preventing quantile crossing across all 7 nominal quantiles $\tau \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$ via cumulative softplus increments.
* **Post-hoc Conformalized Quantile Regression (CQR):** Finite-sample coverage calibration on a CQR calibration on the validation partition, which was also used for early stopping.
---

## 2. Experimental Conditions (C1, C2, C3)

The experimental evaluation is structured into three distinct configurations:

| Condition | Look-back Steps | Look-back Window | Features | Description |
| :---: | :---: | :---: | :---: | :---|
| **C1** | 1 step | 15 minutes | 26 features | Short-context benchmark (single immediate past step) |
| **C2** | 48 steps | 12 hours | 26 features | Full multi-horizon temporal context with all meteorological features |
| **C3** | 48 steps | 12 hours | 10 features | Compact feature subset selected via permutation feature importance |

### C3 Feature Subsets
As reported in Table 3 of `Manuscript`, training-window permutation importance yields the following Top-10 ranking ($\Delta\text{RMSE}$ in MW):
  1. `Actual_MW` ($1799.71$)
  2. `hour` ($20.29$)
  3. `solarenergy` ($8.90$)
  4. `solarradiation` ($8.28$)
  5. `uvindex` ($5.44$)
  6. `windspeed` ($3.90$)
  7. `feelslike` ($2.82$)
  8. `TEMP` ($2.55$)
  9. `sealevelpressure` ($2.01$)
  10. `icon_enc` ($1.90$)

---

## 3. Repository Structure

```
https://github.com/241rcs0005santhosh/Probabilistic-STLF-AP-Grid.git
├── README.md                           # Master documentation & reproduction manual
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Exclusion rules for raw datasets and cache
├── run_all.py                          # Master reproduction orchestrator (Steps 1–9)
├── config/
│   └── config.json                     # Complete experimental configuration & gold fixtures
├── checkpoints/                        # 9 pre-trained .keras model checkpoints
│   ├── C1_LSTM_Q.keras                 # C1 LSTM-Q (25,607 params)
│   ├── C1_CNN_BiGRU_Q.keras            # C1 CNN-BiGRU-Q (72,199 params)
│   ├── C1_DTAQ.keras                   # C1 DTAQ (127,207 params)
│   ├── C2_LSTM_Q.keras                 # C2 LSTM-Q (25,607 params)
│   ├── C2_CNN_BiGRU_Q.keras            # C2 CNN-BiGRU-Q (72,199 params)
│   ├── C2_DTAQ.keras                   # C2 DTAQ (127,207 params)
│   ├── C3_LSTM_Q.keras                 # C3 LSTM-Q (21,511 params)
│   ├── C3_CNN_BiGRU_Q.keras            # C3 CNN-BiGRU-Q (69,127 params)
│   └── C3_DTAQ.keras                   # C3 DTAQ (123,111 params)
├── src/
│   ├── common.py                       # Data loader, chronological splitters, evaluation metrics
│   ├── common_tf.py                    # Custom Keras layers (NCQ Head, Bahdanau Attention, Pinball loss)
│   ├── models.py                       # Standalone model builders (LSTM-Q, CNN-BiGRU-Q, DTAQ)
│   ├── train.py                        # Model training engine from scratch
│   ├── run_c1.py                       # Standalone execution for Experiment C1
│   ├── run_c2.py                       # Standalone execution for Experiment C2
│   ├── run_c3.py                       # Standalone execution for Experiment C3
│   ├── s01_legacy_inference.py         # Step 1: Verification against manuscript gold benchmarks
│   ├── s02_data_audit.py               # Step 2: Timestamp audit, partitions, receptive field audit
│   ├── s03_cqr_diagnostics.py          # Step 3: Conformal diagnostics & prospective calibration
│   ├── s04_hac_dm_baselines.py         # Step 4: Diebold-Mariano HAC tests, baselines, ramps & peaks
│   ├── s05_frozen_sensitivity.py       # Step 5: Repeated actual load (flat run) sensitivity
│   ├── s06_permutation_importance.py   # Step 6: Permutation feature ranking & stability
│   ├── s07_attention_perturbation.py   # Step 7: At-inference attention perturbation diagnostics
│   ├── s08_weather_lag_diagnostic.py   # Step 8: Weather information-set stress tests
│   └── s09_component_ablation.py       # Step 9: Sequential component ablation study
└── results/                            # Generated output directory
---

## 4. Installation & Environment Setup

### Prerequisites
* Python 3.10 or 3.11
* TensorFlow 2.10 through 2.15
* CUDA-compatible GPU (recommended for training; CPU is supported for inference)

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/241rcs0005santhosh/Probabilistic-STLF-AP-Grid.git
cd Probabilistic-STLF-AP-Grid

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or: .\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

The processed dataset is not redistributed in the repository because of the source terms.

### Obtaining Data
* **Grid Demand Telemetry:** Historical 15-minute grid demand can be requested from the Andhra Pradesh State Load Despatch Centre ([APSLDC Portal](https://sldc.aptransco.co.in/)).
* **Meteorological Data:** Surface weather observations (temperature, solar radiation, humidity, etc.) are sourced from the Visual Crossing Weather API for coordinates corresponding to the state load centers.

### File Format Requirements
Save the processed dataset as `APSLDC_15min_Clean.xlsx` in the repository root (or specify its location via `P3_DATA`). The file must contain:
1. `Datetime`: Timestamps at uniform 15-minute intervals.(two absent timestamps (27 Aug 2021, 00:00 and 12:00) are not filled.
2. `Actuals in MW` (or `Actual_MW`): Net system demand in Megawatts.
3. 25 meteorological and temporal feature columns:
   `TEMP`, `feelslike`, `dew`, `humidity`, `precip`, `precipprob`, `windgust`, `windspeed`, `winddir`, `sealevelpressure`, `cloudcover`, `visibility`, `solarradiation`, `solarenergy`, `uvindex`, `severerisk`, `conditions`, `icon`, `preciptype`, `DAY NAME`, `holiday_status`, `hour`, `dow`, `is_weekend`.

### Chronological Splitting Protocol
training 72% 105,165 windows (1 April 2021 12:00 to 31 March 2024 23:30), 
validation 08% 11,685 (31 March 2024 23:45 to 31 July 2024 16:45, used for early stopping and as the CQR calibration set), 
test 20% 29,212 (31 July 2024 17:00 to 31 May 2025 23:45).

---

## 6. Execution Sequence & Reproduction Steps

All experiments can be executed either via the master runner or through step-by-step modular scripts.

### Option A: Master Execution (All Steps)

```bash
# Full reproduction (computes all tables and figures)
python run_all.py

# Quick smoke test mode (reduced bootstrap repeats and quick checks)
python run_all.py --quick
```

### Option B: Step-by-Step Execution

#### Step 1: Reproduction Verification Against Manuscript Benchmarks
Loads all 9 pre-trained checkpoints, performs raw quantile inference on calibration and test sets, computes $q_{\text{hat}}$, and checks every metric against manuscript reference constants:
```bash
python src/s01_legacy_inference.py
```

#### Step 2: Data Audit & Partition Manifest
Inspects timestamp continuity, detects flat runs, audits weather alignment, and computes receptive field from `.keras` JSON configs:
```bash
python src/s02_data_audit.py
```

#### Step 3: Conformal Diagnostics & Prospective Calibration
Calculates signed CQR scores, evaluates prospective split-conformal, Adaptive Conformal Inference (ACI), and Mondrian CQR (season $\times$ peak):
```bash
python src/s03_cqr_diagnostics.py
```

#### Step 4: Diebold-Mariano HAC Tests & Statistical Baselines
Runs Newey-West HAC-adjusted DM tests against persistence, seasonal-naive (24h, 7d), and competitive deep models, with Holm family-wise correction:
```bash
python src/s04_hac_dm_baselines.py
```

#### Step 5: Repeated Sensor Values (Flat Run) Sensitivity
Evaluates forecast stability when consecutive identical load values (sensor freeze artifacts) are filtered out:
```bash
python src/s05_frozen_sensitivity.py
```

#### Step 6: Permutation Feature Importance
Performs shuffle permutation importance on training windows and evaluates feature provenance against manuscript Table 3:
```bash
python src/s06_permutation_importance.py
```

#### Step 7: Attention Perturbation Diagnostics
Perturbs Bahdanau attention weights (uniform, temporal shuffle, top-$k$ mask, last-step mask) to inspect temporal reliance and exports seasonal attention profiles:
```bash
python src/s07_attention_perturbation.py
```

#### Step 8: Weather Latency Stress Tests
Tests model degradation under simulated weather reporting latencies (1-hour lag, 3-hour lag, zeroed weather):
```bash
python src/s08_weather_lag_diagnostic.py
```

#### Step 9: Component Ablation Study
Sequentially evaluates TCN-Point $\to$ TCN-Q $\to$ DTAQ-noCQR $\to$ DTAQ-CQR:
```bash
python src/s09_component_ablation.py
```

---

## 7. Running Individual Experiments (C1, C2, C3)

To inspect or execute each experimental condition independently:

```bash
# Run Experiment C1 (Look-back = 1 step, 26 features)
python src/run_c1.py

# Run Experiment C2 (Look-back = 48 steps, 26 features)
python src/run_c2.py

# Run Experiment C3 (Look-back = 48 steps, 10 features)
python src/run_c3.py
```

---

## 8. Training Models from Scratch

To retrain any model architecture under any experimental condition from scratch:

```bash
# Train DTAQ on Condition C2 (12-hour look-back, 26 features)
python src/train.py --condition C2 --model DTAQ

# Train LSTM-Q on Condition C1
python src/train.py --condition C1 --model LSTM_Q

# Train CNN-BiGRU-Q on Condition C3
python src/train.py --condition C3 --model CNN_BiGRU_Q

# Retrain all 9 combinations
python src/train.py --all
```

Hyperparameters are strictly preserved:
* **Batch size:** 256
* **Learning rate:** Cosine decay restarts ($10^{-3} \to 10^{-5}$)
* **Loss:** 7-quantile average pinball loss
* **Patience:** 5 epochs (LSTM-Q, CNN-BiGRU-Q), 15 epochs (DTAQ)
* **Random seed:** reported numbers use seed 42

---

## 9. Benchmark Verification (Manuscript Values)

The script `s01_legacy_inference.py` validates that the existing checkpoints reproduce the manuscript results within strict numerical tolerance:

| Condition | Model | RMSE (MW) | MAE (MW) | MAPE (%) | $R^2$ | PICP (Raw) | PICP (CQR) | $q_{\text{hat}}$ (MW) | Parameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | LSTM-Q | 217.28 | 112.20 | 1.2843 | 0.9746 | 0.8961 | 0.8925 | -1.80 | 25,607 |
| **C1** | CNN-BiGRU-Q | 216.69 | 108.67 | 1.2425 | 0.9747 | 0.9481 | 0.9155 | -47.12 | 72,199 |
| **C1** | **DTAQ** | **214.58** | **109.20** | **1.2416** | **0.9752** | **0.9032** | **0.9165** | **+14.84** | **127,207** |
| **C2** | LSTM-Q | 206.54 | 104.51 | 1.2079 | 0.9771 | 0.9056 | 0.8851 | -11.57 | 25,607 |
| **C2** | CNN-BiGRU-Q | 199.99 | 105.17 | 1.2087 | 0.9785 | 0.9378 | 0.9126 | -23.05 | 72,199 |
| **C2** | **DTAQ** | **173.88** | **77.16** | **0.9035** | **0.9837** | **0.8791** | **0.8926** | **+6.96** | **127,207** |
| **C3** | LSTM-Q | 207.11 | 98.13 | 1.1301 | 0.9769 | 0.9138 | 0.8951 | -12.61 | 21,511 |
| **C3** | CNN-BiGRU-Q | 189.34 | 90.48 | 1.0390 | 0.9807 | 0.9424 | 0.9234 | -21.05 | 69,127 |
| **C3** | **DTAQ** | **173.82** | **75.55** | **0.8860** | **0.9837** | **0.8382** | **0.8824** | **+17.75** | **123,111** |

---

## 10. Citation

If you use this code or model in your research, please cite:

```bibtex
@article{santhosh2026dtaq,
  title={Probabilistic Short-Term Load Forecasting with Dilated Temporal Convolutional Networks and Conformal Calibration: Evidence from the Andhra Pradesh State Grid},
  author={K N S K Santhosh and R Tamilkodi and A Ramesh},
  journal={International Journal of Intelligent Engineering and Systems},
  year={2026}
}
```

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
