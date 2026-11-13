"""
train.py
========
Model training script for all architectures and conditions.

Supports:
  - Models: LSTM_Q, CNN_BiGRU_Q, DTAQ
  - Conditions: C1 (look_back=1, 26 feats), C2 (look_back=48, 26 feats), C3 (look_back=48, 10 feats)
  - Loss: 7-quantile pinball loss (averaging taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
  - Optimizer: Adam with CosineDecayRestarts (initial_lr=1e-3, first_decay_steps=10*steps_per_epoch, alpha=1e-5)
  - Early stopping: monitor='val_loss', patience=(5 for LSTM/CNN-BiGRU, 15 for DTAQ)
  - Random seed: 42 (reproducible initialization)

Usage:
    python src/train.py --condition C2 --model DTAQ
    python src/train.py --condition C1 --model LSTM_Q
    python src/train.py --all
"""

import os
import sys
import argparse
import gc
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from common_tf import *
from models import build_lstm_q, build_cnn_bigru_q, build_dtaq


def train_single_model(cond, arch, save_dir=None):
    if save_dir is None:
        save_dir = CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"TRAINING: Condition={cond}, Model={arch}")
    print("=" * 65)

    df, feats, ap = load_frame()
    feat_set = TOP10_LEGACY if cond == "C3" else feats
    lb = LB_OF[cond]

    p = prepare(df, feat_set, lb, protocol="legacy")
    Xtr, ytr = p.X[p.idx["train"]], p.y[p.idx["train"]]
    Xvl, yvl = p.X[p.idx["cal"]],   p.y[p.idx["cal"]]
    Xte, yte = p.X[p.idx["test"]],  p.y[p.idx["test"]]

    input_shape = (lb, len(feat_set))
    batch_size = 256
    steps_per_epoch = max(1, int(np.ceil(len(Xtr) / batch_size)))

    reseed(SEED)
    keras.backend.clear_session()
    gc.collect()

    if arch == "LSTM_Q":
        model = build_lstm_q(input_shape, lr=1e-3, steps_per_epoch=steps_per_epoch)
    elif arch == "CNN_BiGRU_Q":
        model = build_cnn_bigru_q(input_shape, lr=1e-3, steps_per_epoch=steps_per_epoch)
    elif arch == "DTAQ":
        model, _ = build_dtaq(input_shape, filters=64, lr=1e-3, steps_per_epoch=steps_per_epoch)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    out_ckpt = os.path.join(save_dir, f"{cond}_{arch}.keras")
    patience = PATIENCE[arch]
    max_epochs = MAX_EPOCHS[arch]

    callbacks = [
        keras.callbacks.ModelCheckpoint(out_ckpt, monitor="val_loss", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        _Tick(f"{cond}_{arch}", every=5)
    ]

    print(f"Input shape: {input_shape} | Parameters: {model.count_params():,}")
    print(f"Training samples: {len(Xtr):,} | Val samples: {len(Xvl):,} | Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs} | Patience: {patience}")

    t0 = time.time()
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xvl, yvl),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    elapsed = round(time.time() - t0, 1)

    print(f"Training finished in {elapsed}s.")
    print(f"Checkpoint saved to: {out_ckpt}")

    # Evaluate on test set
    yq_test = q_to_mw(p, predict_q(model, Xte))
    y_test  = inv(p, yte)
    yq_cal  = q_to_mw(p, predict_q(model, Xvl))
    y_cal   = inv(p, yvl)

    qh = cqr_qhat(cqr_scores(y_cal, yq_cal))
    summary = summarize(f"{cond}_{arch}", y_test, yq_test, qh, dict(
        condition=cond, model=arch, params=int(model.count_params()),
        epochs_trained=len(history.history["loss"]), training_seconds=elapsed
    ))
    print(f"Test RMSE: {summary['RMSE']:.2f} MW | PICP (raw): {summary['PICP_raw']:.4f} | "
          f"PICP (CQR): {summary['PICP_cqr']:.4f} | q_hat: {qh:+.2f} MW")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train DTAQ, LSTM-Q, and CNN-BiGRU-Q models.")
    parser.add_argument("--condition", choices=["C1", "C2", "C3"], default="C2",
                        help="Experimental condition: C1 (1 lag, 26 feats), C2 (48 lags, 26 feats), C3 (48 lags, 10 feats).")
    parser.add_argument("--model", choices=["LSTM_Q", "CNN_BiGRU_Q", "DTAQ"], default="DTAQ",
                        help="Model architecture.")
    parser.add_argument("--all", action="store_true",
                        help="Train all 9 combinations (3 conditions x 3 models).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save trained checkpoints.")
    args = parser.parse_args()

    if args.all:
        for c in CONDS:
            for a in ARCHS:
                train_single_model(c, a, save_dir=args.save_dir)
    else:
        train_single_model(args.condition, args.model, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
