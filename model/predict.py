"""
Prediction / Inference Script – DVB-S2 Channel Estimation
==========================================================
Loads trained model weights and runs inference on test data to predict
future SNR trajectories.  Includes MODCOD selection via ETSI lookup,
evaluation metrics, and visualization.

Usage
-----
    python predict.py                                   # defaults
    python predict.py --weights weights/best.weights.h5 # specific checkpoint
    python predict.py --plot                            # generate plots
    python predict.py --sample_idx 42                   # visualise one sample
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np

# Ensure model.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Lazy imports (avoid slow TF init until needed)
# ============================================================================
def import_tf_and_model():
    """Import TensorFlow and model builders (deferred for fast --help)."""
    import tensorflow as tf
    import keras
    from model import build_stage2_only_model, build_two_stage_model, select_modcod
    return tf, keras, build_stage2_only_model, build_two_stage_model, select_modcod


# ============================================================================
# Data Loading
# ============================================================================
def load_test_data(data_dir):
    """Load the test split from preprocessed data."""
    data_dir = Path(data_dir)
    test_path = data_dir / "stage2_test.npz"

    if not test_path.exists():
        print(f"  ERROR: Test data not found at {test_path}")
        print("  Run preprocess.py first to generate the dataset.")
        sys.exit(1)

    data = np.load(test_path)
    return data["X"], data["Y"]


def load_full_test_data(data_dir):
    """Load the test split for the full model from preprocessed data."""
    data_dir = Path(data_dir)
    test_path = data_dir / "full_test.npz"

    if not test_path.exists():
        print(f"  ERROR: Full-model test data not found at {test_path}")
        print("  Make sure to run preprocess.py with '--mode full'.")
        sys.exit(1)

    data = np.load(test_path)
    return data["pilot_blocks"], data["pilot_masks"], data["modcod"], data["Y"]


def load_norm_stats(data_dir):
    """Load normalization statistics (for un-normalizing predictions if needed)."""
    stats_path = Path(data_dir) / "norm_stats.npz"
    if stats_path.exists():
        data = np.load(stats_path)
        return {"mean": data["mean"], "std": data["std"]}
    return None


# ============================================================================
# Evaluation Metrics
# ============================================================================
def compute_metrics(y_true, y_pred):
    """Compute regression metrics between true and predicted SNR.

    Parameters
    ----------
    y_true : (n_samples, H)  ground-truth SNR (dB)
    y_pred : (n_samples, H)  predicted SNR (dB)

    Returns
    -------
    metrics : dict
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    # Per-step metrics
    mse_per_step = np.mean((y_true - y_pred) ** 2, axis=0)
    mae_per_step = np.mean(np.abs(y_true - y_pred), axis=0)

    # Correlation
    flat_true = y_true.flatten()
    flat_pred = y_pred.flatten()
    if np.std(flat_true) > 0 and np.std(flat_pred) > 0:
        correlation = np.corrcoef(flat_true, flat_pred)[0, 1]
    else:
        correlation = 0.0

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "correlation": float(correlation),
        "mse_per_step": mse_per_step.tolist(),
        "mae_per_step": mae_per_step.tolist(),
    }


# ============================================================================
# MODCOD Selection for Predicted Trajectories
# ============================================================================
def predict_modcod_trajectory(snr_trajectory, margin_db=1.0):
    """Select MODCOD for each step in a predicted SNR trajectory.

    Parameters
    ----------
    snr_trajectory : (H,) predicted SNR in dB
    margin_db : float

    Returns
    -------
    modcods : list of dict, length H
    """
    from model import select_modcod
    return [select_modcod(float(snr), margin_db) for snr in snr_trajectory]


# ============================================================================
# Visualization
# ============================================================================
def plot_predictions(y_true, y_pred, sample_indices, output_dir, delta_t=0.1):
    """Plot predicted vs actual SNR trajectories for selected samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not installed, skipping plots.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    H = y_true.shape[1]
    time_axis = np.arange(1, H + 1) * delta_t  # future time in seconds

    # --- Individual sample plots ---
    for idx in sample_indices:
        if idx >= len(y_true):
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_axis, y_true[idx], "b-o", markersize=3,
                label="Ground Truth", linewidth=1.5)
        ax.plot(time_axis, y_pred[idx], "r--s", markersize=3,
                label="Predicted", linewidth=1.5, alpha=0.8)
        ax.fill_between(time_axis,
                        y_pred[idx] - 1.0, y_pred[idx] + 1.0,
                        alpha=0.15, color="red", label="+/- 1 dB margin")

        ax.set_xlabel("Future Time (s)", fontsize=12)
        ax.set_ylabel("SNR (dB)", fontsize=12)
        ax.set_title(f"SNR Prediction - Sample #{idx}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        path = output_dir / f"prediction_sample_{idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Error distribution plot ---
    errors = y_pred - y_true  # (n_samples, H)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of errors
    axes[0].hist(errors.flatten(), bins=80, color="steelblue",
                 edgecolor="white", alpha=0.8, density=True)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Prediction Error (dB)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Error Distribution", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Per-step RMSE
    rmse_per_step = np.sqrt(np.mean(errors ** 2, axis=0))
    axes[1].bar(time_axis, rmse_per_step, width=delta_t * 0.8,
                color="darkorange", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Future Time (s)", fontsize=12)
    axes[1].set_ylabel("RMSE (dB)", fontsize=12)
    axes[1].set_title("RMSE per Prediction Step", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "error_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- MODCOD selection comparison ---
    from model import select_modcod

    fig, ax = plt.subplots(figsize=(12, 5))
    sample_idx = sample_indices[0] if sample_indices else 0
    if sample_idx < len(y_true):
        true_snr = y_true[sample_idx]
        pred_snr = y_pred[sample_idx]

        true_modcods = [select_modcod(float(s), 1.0) for s in true_snr]
        pred_modcods = [select_modcod(float(s), 1.0) for s in pred_snr]

        true_req = [m["required_esn0_db"] for m in true_modcods]
        pred_req = [m["required_esn0_db"] for m in pred_modcods]

        ax.step(time_axis, true_req, "b-", where="mid",
                label="True MODCOD threshold", linewidth=2)
        ax.step(time_axis, pred_req, "r--", where="mid",
                label="Predicted MODCOD threshold", linewidth=2)
        ax.plot(time_axis, true_snr, "b.", alpha=0.4, markersize=4,
                label="True SNR")
        ax.plot(time_axis, pred_snr, "r.", alpha=0.4, markersize=4,
                label="Predicted SNR")

        ax.set_xlabel("Future Time (s)", fontsize=12)
        ax.set_ylabel("SNR / Threshold (dB)", fontsize=12)
        ax.set_title(f"MODCOD Selection - Sample #{sample_idx}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    path = output_dir / "modcod_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main Prediction Pipeline
# ============================================================================
def predict(args):
    """Run the full prediction / evaluation pipeline."""
    print("=" * 70)
    print("  DVB-S2 Channel Prediction & Evaluation")
    print("=" * 70)

    # --- Import TF and model ---
    tf, keras, build_stage2_only_model, build_two_stage_model, select_modcod = \
        import_tf_and_model()

    # Load training config if available (do this first to know the model type)
    weights_dir = Path(args.weights_dir)
    train_config_path = weights_dir / "train_config.json"
    if train_config_path.exists():
        with open(train_config_path) as f:
            train_cfg = json.load(f)
        print(f"       Train config loaded: {train_config_path}")
    else:
        train_cfg = {}

    model_type = train_cfg.get("model_type", args.model)

    # --- Load test data ---
    print("\n[1/5] Loading test data ...")
    data_dir = Path(args.data_dir)
    if model_type == "stage2":
        X_test, Y_test = load_test_data(data_dir)
        print(f"       X_test shape: {X_test.shape}")
        print(f"       Y_test shape: {Y_test.shape}")

        # Infer dimensions
        history_len = X_test.shape[1]
        d_in = X_test.shape[2]
        prediction_horizon = Y_test.shape[1]
        X_predict = X_test
    else:
        pb_test, pm_test, mc_test, Y_test = load_full_test_data(data_dir)
        print(f"       Pilot blocks shape : {pb_test.shape}")
        print(f"       Pilot masks shape  : {pm_test.shape}")
        print(f"       MODCOD shape       : {mc_test.shape}")
        print(f"       Y_test shape       : {Y_test.shape}")

        history_len = pb_test.shape[1]
        d_in = 7
        prediction_horizon = Y_test.shape[1]
        X_test = pb_test
        X_predict = [pb_test, pm_test, mc_test]

    # --- Build model and load weights ---
    print("\n[2/5] Building model and loading weights ...")

    if model_type == "stage2":
        model = build_stage2_only_model(
            d_in=train_cfg.get("d_in", d_in),
            history_len=train_cfg.get("history_len", history_len),
            prediction_horizon=train_cfg.get("prediction_horizon",
                                              prediction_horizon),
            hidden_units=train_cfg.get("hidden_units", 128),
            n_lstm_layers=train_cfg.get("n_lstm_layers", 2),
            dropout_rate=train_cfg.get("dropout", 0.1),
            l2_reg=train_cfg.get("l2_reg", 1e-4),
        )
    else:
        model = build_two_stage_model(
            history_len=train_cfg.get("history_len", history_len),
            prediction_horizon=train_cfg.get("prediction_horizon",
                                              prediction_horizon),
            hidden_units=train_cfg.get("hidden_units", 128),
            n_lstm_layers=train_cfg.get("n_lstm_layers", 2),
            dropout_rate=train_cfg.get("dropout", 0.1),
            l2_reg=train_cfg.get("l2_reg", 1e-4),
        )

    # Load weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"\n  ERROR: Weights file not found: {weights_path}")
        print("  Run train.py first to train the model.")
        sys.exit(1)

    model.load_weights(str(weights_path))
    print(f"       Weights loaded from: {weights_path}")
    print(f"       Model: {model.name}  |  Params: {model.count_params():,}")

    # --- Run Prediction ---
    print(f"\n[3/5] Running predictions on {X_test.shape[0]} test samples ...")
    Y_pred = model.predict(X_predict, batch_size=args.batch_size, verbose=1)
    print(f"       Predictions shape: {Y_pred.shape}")

    # --- Compute Metrics ---
    print("\n[4/5] Computing evaluation metrics ...")
    metrics = compute_metrics(Y_test, Y_pred)

    print(f"\n  {'Metric':<25s} {'Value':>12s}")
    print(f"  {'-'*25} {'-'*12}")
    print(f"  {'MSE (dB^2)':<25s} {metrics['mse']:>12.6f}")
    print(f"  {'MAE (dB)':<25s} {metrics['mae']:>12.4f}")
    print(f"  {'RMSE (dB)':<25s} {metrics['rmse']:>12.4f}")
    print(f"  {'Correlation':<25s} {metrics['correlation']:>12.4f}")

    # Per-step breakdown (first 5 and last 5)
    H = prediction_horizon
    data_cfg_path = data_dir / "config.json"
    delta_t = 0.1
    if data_cfg_path.exists():
        with open(data_cfg_path) as f:
            delta_t = json.load(f).get("delta_t", 0.1)

    print(f"\n  Per-step RMSE (first & last 5 of {H} steps):")
    rmse_steps = np.sqrt(metrics["mse_per_step"])
    for i in list(range(min(5, H))) + ["..."] + list(range(max(H-5, 5), H)):
        if i == "...":
            print(f"    {'...'}")
        else:
            t_s = (i + 1) * delta_t
            print(f"    Step {i+1:>3d} (t+{t_s:.1f}s): "
                  f"RMSE = {rmse_steps[i]:.4f} dB")

    # --- MODCOD Selection for a sample ---
    print("\n  MODCOD Selection Example (Sample #0):")
    if len(Y_pred) > 0:
        modcods = predict_modcod_trajectory(Y_pred[0], margin_db=args.margin_db)
        print(f"  {'Step':>4s}  {'Pred SNR':>9s}  {'True SNR':>9s}  "
              f"{'MODCOD':>10s}  {'Rate':>5s}  {'Req dB':>7s}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*5}  {'-'*7}")
        for i in range(min(10, H)):
            mc = modcods[i]
            print(f"  {i+1:>4d}  {Y_pred[0, i]:>9.2f}  {Y_test[0, i]:>9.2f}  "
                  f"{mc['modulation']:>10s}  {mc['code_rate']:>5s}  "
                  f"{mc['required_esn0_db']:>7.1f}")
        if H > 10:
            print(f"  {'...':>4s}")

    # --- Save Results ---
    print("\n[5/5] Saving results ...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predictions
    np.savez_compressed(
        output_dir / "predictions.npz",
        Y_true=Y_test, Y_pred=Y_pred,
    )

    # Metrics
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in metrics.items()}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"       Metrics saved: {output_dir / 'metrics.json'}")

    # Plots
    if args.plot:
        print("\n  Generating plots ...")
        sample_indices = args.sample_indices if args.sample_indices else [0, 1, 2]
        plot_predictions(Y_test, Y_pred, sample_indices, output_dir, delta_t)

    print("\n" + "=" * 70)
    print(f"  Prediction complete!  Results in: {output_dir.resolve()}")
    print("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Run predictions with trained DVB-S2 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["stage2", "full"], default="stage2",
                   help="Model variant (overridden by train_config.json if found).")
    p.add_argument("--weights", type=str, default="weights/best.weights.h5",
                   help="Path to trained model weights (.weights.h5).")
    p.add_argument("--weights_dir", type=str, default="weights",
                   help="Directory containing train_config.json.")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Path to preprocessed data directory.")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Directory to save prediction results.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Inference batch size.")
    p.add_argument("--margin_db", type=float, default=1.0,
                   help="Implementation margin (dB) for MODCOD selection.")
    p.add_argument("--plot", action="store_true",
                   help="Generate visualization plots.")
    p.add_argument("--sample_indices", type=int, nargs="*", default=None,
                   help="Sample indices to plot (e.g. --sample_indices 0 10 50).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)
