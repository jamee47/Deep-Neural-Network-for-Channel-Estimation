"""
Training Script – Two-Stage CNN + Bi-LSTM for DVB-S2 Channel Estimation
=========================================================================
Trains either the full two-stage model or the stage-2-only Bi-LSTM model
on preprocessed DVB-S2 channel data.

Features
--------
- Automatic GPU detection and memory-growth configuration
- Mixed-precision training (optional, for faster GPU training)
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
- Saves model weights, training history, and a summary plot
- Supports resuming from a previous checkpoint

Usage
-----
    python train.py                                 # defaults (stage-2)
    python train.py --model full                    # full two-stage model
    python train.py --epochs 200 --batch_size 64    # custom training
    python train.py --resume weights/best.weights.h5  # resume training
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
import keras

# Ensure model.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_two_stage_model, build_stage2_only_model


# ============================================================================
# GPU Configuration
# ============================================================================
def setup_gpu():
    """Detect and configure GPU with memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  GPU(s) detected: {len(gpus)}")
        for gpu in gpus:
            print(f"    -> {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"       Memory growth setting failed: {e}")
        # Use first GPU
        print(f"  Training will use GPU: {gpus[0].name}")
    else:
        print("\n  No GPU detected. Training will use CPU.")
        print("  (This will be significantly slower for LSTM models)")
    return gpus


# ============================================================================
# Data Loading
# ============================================================================
def load_stage2_data(data_dir):
    """Load pre-extracted stage-2 features from .npz files."""
    data_dir = Path(data_dir)

    train = np.load(data_dir / "stage2_train.npz")
    val = np.load(data_dir / "stage2_val.npz")

    return {
        "X_train": train["X"], "Y_train": train["Y"],
        "X_val": val["X"], "Y_val": val["Y"],
    }


def create_tf_dataset(X, Y, batch_size, shuffle=True, buffer_size=10000):
    """Create a tf.data.Dataset with optional shuffling and prefetching."""
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================================
# Callbacks
# ============================================================================
def build_callbacks(output_dir, patience_es=20, patience_lr=7):
    """Create training callbacks.

    - ModelCheckpoint: save best weights (by val_loss)
    - EarlyStopping: stop if val_loss doesn't improve
    - ReduceLROnPlateau: halve LR on plateau
    - CSVLogger: log metrics per epoch to CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / "training_log.csv"),
            append=True,
        ),
    ]

    return callbacks


# ============================================================================
# Training History Plot
# ============================================================================
def save_training_plots(history, output_dir):
    """Save loss and metric plots to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not installed, skipping plots.")
        return

    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    if "mae" in history.history:
        axes[1].plot(history.history["mae"], label="Train MAE")
        axes[1].plot(history.history["val_mae"], label="Val MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (dB)")
        axes[1].set_title("Mean Absolute Error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # RMSE
    if "rmse" in history.history:
        axes[2].plot(history.history["rmse"], label="Train RMSE")
        axes[2].plot(history.history["val_rmse"], label="Val RMSE")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("RMSE (dB)")
        axes[2].set_title("Root Mean Squared Error")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Training plots saved to: {plot_path}")


# ============================================================================
# Main Training Loop
# ============================================================================
def train(args):
    """Execute the training pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- GPU Setup --------------------------------------------------------
    gpus = setup_gpu()

    print("\n" + "=" * 70)
    print(f"  Training Configuration")
    print("=" * 70)
    print(f"  Model type     : {args.model}")
    print(f"  Data directory  : {args.data_dir}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.learning_rate}")
    print(f"  Hidden units    : {args.hidden_units}")
    print(f"  LSTM layers     : {args.n_lstm_layers}")
    print(f"  Dropout         : {args.dropout}")
    print(f"  L2 reg          : {args.l2_reg}")
    print(f"  Device          : {'GPU' if gpus else 'CPU'}")
    print("=" * 70)

    # --- Load Data --------------------------------------------------------
    print("\n[1/4] Loading data ...")
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"\n  ERROR: Data directory '{data_dir}' not found.")
        print("  Run preprocess.py first to generate the dataset.")
        sys.exit(1)

    data = load_stage2_data(data_dir)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val = data["X_val"], data["Y_val"]

    print(f"       Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"       Val  : X={X_val.shape}, Y={Y_val.shape}")

    # Infer dimensions from data
    history_len = X_train.shape[1]
    d_in = X_train.shape[2]
    prediction_horizon = Y_train.shape[1]

    # Load config if available
    config_path = data_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data_cfg = json.load(f)
        print(f"       Config loaded from: {config_path}")
    else:
        data_cfg = {}

    # Create tf.data pipelines
    train_ds = create_tf_dataset(X_train, Y_train, args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, Y_val, args.batch_size, shuffle=False)

    # --- Build Model ------------------------------------------------------
    print("\n[2/4] Building model ...")
    if args.model == "stage2":
        model = build_stage2_only_model(
            d_in=d_in,
            history_len=history_len,
            prediction_horizon=prediction_horizon,
            hidden_units=args.hidden_units,
            n_lstm_layers=args.n_lstm_layers,
            dropout_rate=args.dropout,
            l2_reg=args.l2_reg,
            learning_rate=args.learning_rate,
        )
    elif args.model == "full":
        num_blocks = data_cfg.get("num_blocks", 16)
        n_max = data_cfg.get("n_max", 64)
        c_pilot = data_cfg.get("c_pilot", 2)
        d_f = data_cfg.get("d_f", 64)
        d_s = data_cfg.get("d_s", 5)

        model = build_two_stage_model(
            num_blocks=num_blocks,
            n_max=n_max,
            c_pilot=c_pilot,
            d_f=d_f,
            d_s=d_s,
            history_len=history_len,
            prediction_horizon=prediction_horizon,
            hidden_units=args.hidden_units,
            n_lstm_layers=args.n_lstm_layers,
            dropout_rate=args.dropout,
            l2_reg=args.l2_reg,
            learning_rate=args.learning_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.summary(line_length=100)

    # Resume from checkpoint
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        model.load_weights(args.resume)

    # --- Prepare output directory -----------------------------------------
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # --- Callbacks --------------------------------------------------------
    callbacks = build_callbacks(
        output_dir=weights_dir,
        patience_es=args.patience,
        patience_lr=max(args.patience // 3, 3),
    )

    # --- Train ------------------------------------------------------------
    print(f"\n[3/4] Training for up to {args.epochs} epochs ...")
    print("-" * 70)

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"  Training completed in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")

    # --- Save Final Weights & Artifacts -----------------------------------
    print("\n[4/4] Saving model artifacts ...")

    # Save final weights (in addition to best checkpoint)
    final_weights_path = weights_dir / "final.weights.h5"
    model.save_weights(str(final_weights_path))
    print(f"       Final weights : {final_weights_path}")

    # Save training history as JSON
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    history_path = weights_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"       History JSON  : {history_path}")

    # Save training config
    train_config = {
        "model_type": args.model,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_units": args.hidden_units,
        "n_lstm_layers": args.n_lstm_layers,
        "dropout": args.dropout,
        "l2_reg": args.l2_reg,
        "d_in": d_in,
        "history_len": history_len,
        "prediction_horizon": prediction_horizon,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "timestamp": timestamp,
        "device": "GPU" if gpus else "CPU",
    }
    config_out = weights_dir / "train_config.json"
    with open(config_out, "w") as f:
        json.dump(train_config, f, indent=2)
    print(f"       Train config  : {config_out}")

    # Plot training curves
    save_training_plots(history, weights_dir)

    # Final summary
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val = min(history.history["val_loss"])
    print("\n" + "=" * 70)
    print(f"  Training Summary")
    print("=" * 70)
    print(f"  Best val_loss  : {best_val:.6f}  (epoch {best_epoch})")
    print(f"  Best val_mae   : {min(history.history.get('val_mae', [0])):.4f} dB")
    print(f"  Best val_rmse  : {min(history.history.get('val_rmse', [0])):.4f} dB")
    print(f"  Weights saved  : {weights_dir / 'best.weights.h5'}")
    print("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Train the DVB-S2 Channel Estimation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["stage2", "full"], default="stage2",
                   help="Model variant to train.")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Path to preprocessed data directory.")
    p.add_argument("--weights_dir", type=str, default="weights",
                   help="Directory to save model weights and artifacts.")
    p.add_argument("--epochs", type=int, default=100,
                   help="Maximum number of training epochs.")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Training batch size.")
    p.add_argument("--learning_rate", type=float, default=1e-3,
                   help="Initial Adam learning rate.")
    p.add_argument("--hidden_units", type=int, default=128,
                   help="LSTM hidden units per direction.")
    p.add_argument("--n_lstm_layers", type=int, default=2,
                   help="Number of stacked Bi-LSTM layers.")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="LSTM dropout rate.")
    p.add_argument("--l2_reg", type=float, default=1e-4,
                   help="L2 regularization coefficient.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to weights file to resume training from.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
