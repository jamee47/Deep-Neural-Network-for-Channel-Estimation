"""
Dataset Preprocessor – DVB-S2 MATLAB CSV to Model-Ready Format
================================================================
Loads a CSV dataset exported from MATLAB containing per-frame pilot
symbols and channel metadata, then transforms it into the format
expected by the Two-Stage CNN + Bi-LSTM model.

Expected CSV columns
--------------------
    pilot_re_1 ... pilot_re_P   : Real part of P received pilot symbols
    pilot_im_1 ... pilot_im_P   : Imaginary part of P received pilot symbols
    H_true_re                    : True channel response (real)
    H_true_im                    : True channel response (imaginary)
    snr_dB                       : SNR in dB
    nVar                         : Noise variance (linear)
    modcod                       : MODCOD index
    rainAtt_dB                   : Rain attenuation (dB)

The script produces two output modes:
    1. **stage2** : Pre-extracted channel features -> Bi-LSTM model
    2. **full**   : Raw pilot blocks + masks -> full CNN + Bi-LSTM model

Usage
-----
    python preprocess.py                                     # defaults
    python preprocess.py --csv_path path/to/data.csv         # your CSV
    python preprocess.py --mode full                         # pilot blocks
    python preprocess.py --history_len 200 --horizon 50      # window params
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION – Update path to match your setup
# ============================================================================
DEFAULT_CSV_PATH = r"E:\AVIONICS\Thesis\DNN\data\channel_dataset.csv"


# ============================================================================
# MODCOD Index Lookup (DVB-S2 standard)
# ============================================================================
# Maps MODCOD index -> (modulation_order, code_rate_float)
# Update this if your MATLAB uses a different numbering scheme.
MODCOD_INDEX_MAP = {
    1:  (4,  0.25),    # QPSK 1/4
    2:  (4,  0.50),    # QPSK 1/2
    3:  (4,  0.75),    # QPSK 3/4
    4:  (4,  0.667),   # QPSK 2/3
    5:  (8,  0.75),    # 8PSK 3/4
    6:  (8,  0.833),   # 8PSK 5/6
    7:  (8,  0.667),   # 8PSK 2/3
    8:  (16, 0.75),    # 16APSK 3/4
    9:  (16, 0.833),   # 16APSK 5/6
    10: (16, 0.75),    # 16APSK 3/4 (higher)
    11: (32, 0.75),    # 32APSK 3/4
    12: (32, 0.833),   # 32APSK 5/6
    13: (32, 0.889),   # 32APSK 8/9
    14: (32, 0.90),    # 32APSK 9/10
}


def modcod_index_to_features(index):
    """Convert MODCOD index to (modulation_order, code_rate)."""
    idx = int(round(index))
    if idx in MODCOD_INDEX_MAP:
        return MODCOD_INDEX_MAP[idx]
    # Fallback: return (index, 0) so the model still gets a signal
    return (float(idx), 0.0)


# ============================================================================
# CSV Loader
# ============================================================================
def load_csv(csv_path, delimiter=","):
    """Load the CSV file exported from MATLAB."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"\n  ERROR: CSV file not found: {csv_path}")
        print("  Update DEFAULT_CSV_PATH or use --csv_path argument.")
        sys.exit(1)

    print(f"  Loading: {csv_path}")
    df = pd.read_csv(csv_path, delimiter=delimiter)
    df.columns = df.columns.str.strip()

    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return df


# ============================================================================
# Column Detection & Parsing
# ============================================================================
def detect_pilot_columns(df):
    """Auto-detect pilot_re_* and pilot_im_* columns and their count."""
    re_cols = sorted(
        [c for c in df.columns if c.startswith("pilot_re_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    im_cols = sorted(
        [c for c in df.columns if c.startswith("pilot_im_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    if len(re_cols) != len(im_cols):
        print(f"  WARNING: Mismatched pilot columns: "
              f"{len(re_cols)} Re vs {len(im_cols)} Im")

    n_pilots = min(len(re_cols), len(im_cols))
    print(f"  Detected {n_pilots} pilot symbols per frame "
          f"(Re + Im = {n_pilots * 2} columns)")
    return re_cols[:n_pilots], im_cols[:n_pilots], n_pilots


def detect_metadata_columns(df):
    """Check which metadata columns exist in the CSV."""
    expected = ["H_true_re", "H_true_im", "snr_dB", "nVar", "modcod", "rainAtt_dB"]
    found = {}
    for col in expected:
        # Case-insensitive matching
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if matches:
            found[col] = matches[0]
            print(f"    {col:15s} -> '{matches[0]}'")
        else:
            found[col] = None
            print(f"    {col:15s} -> NOT FOUND (will derive or fill zeros)")
    return found


# ============================================================================
# Pilot Block Builder (for full two-stage model)
# ============================================================================
def build_pilot_blocks(df, re_cols, im_cols, n_pilots, num_blocks, n_max):
    """Group raw pilot symbols into B blocks, pad to N_max, create masks.

    Parameters
    ----------
    df : DataFrame
    re_cols, im_cols : lists of column names
    n_pilots : int  – total pilots per frame
    num_blocks : int – B
    n_max : int – max pilots per block after padding

    Returns
    -------
    blocks : (N_frames, B, N_max, 2) float32  – [Re, Im] per pilot
    masks  : (N_frames, B, N_max)    float32  – 1=valid, 0=padded
    """
    N = len(df)

    # Read all pilot values as arrays
    pilot_re = df[re_cols].values.astype(np.float32)  # (N, n_pilots)
    pilot_im = df[im_cols].values.astype(np.float32)  # (N, n_pilots)

    # Determine how many pilots go into each block
    base_size = n_pilots // num_blocks
    remainder = n_pilots % num_blocks
    # First 'remainder' blocks get (base_size+1) pilots, rest get base_size
    block_sizes = [base_size + (1 if b < remainder else 0)
                   for b in range(num_blocks)]

    actual_max = max(block_sizes)
    if n_max < actual_max:
        print(f"  WARNING: n_max ({n_max}) < largest block ({actual_max}). "
              f"Increasing n_max to {actual_max}.")
        n_max = actual_max

    blocks = np.zeros((N, num_blocks, n_max, 2), dtype=np.float32)
    masks = np.zeros((N, num_blocks, n_max), dtype=np.float32)

    offset = 0
    for b in range(num_blocks):
        sz = block_sizes[b]
        blocks[:, b, :sz, 0] = pilot_re[:, offset:offset + sz]
        blocks[:, b, :sz, 1] = pilot_im[:, offset:offset + sz]
        masks[:, b, :sz] = 1.0
        offset += sz

    print(f"  Pilot blocks: {blocks.shape}  "
          f"(B={num_blocks}, N_max={n_max}, pilots/block={block_sizes[0]}-{block_sizes[-1]})")
    print(f"  Masks: {masks.shape}")
    return blocks, masks, n_max


# ============================================================================
# Channel Feature Extractor (analytical – for stage-2 model)
# ============================================================================
def extract_channel_features(df, meta_cols):
    """Derive the d_s=5 channel-state vector from CSV metadata.

    Features: [channel_gain, attenuation_dB, phase_rad, snr_dB, noise_var]

    Parameters
    ----------
    df : DataFrame
    meta_cols : dict mapping expected names to actual column names

    Returns
    -------
    features : (N, 5) float32
    """
    N = len(df)
    features = np.zeros((N, 5), dtype=np.float32)

    # Channel gain |H| = sqrt(Re^2 + Im^2)
    if meta_cols["H_true_re"] and meta_cols["H_true_im"]:
        h_re = df[meta_cols["H_true_re"]].values.astype(np.float32)
        h_im = df[meta_cols["H_true_im"]].values.astype(np.float32)
        features[:, 0] = np.sqrt(h_re ** 2 + h_im ** 2)
    else:
        print("  WARNING: H_true not found, channel gain set to 1.0")
        features[:, 0] = 1.0

    # Attenuation (dB)
    if meta_cols["rainAtt_dB"]:
        features[:, 1] = df[meta_cols["rainAtt_dB"]].values.astype(np.float32)

    # Phase (rad) = atan2(Im, Re)
    if meta_cols["H_true_re"] and meta_cols["H_true_im"]:
        features[:, 2] = np.arctan2(h_im, h_re).astype(np.float32)

    # SNR (dB)
    if meta_cols["snr_dB"]:
        features[:, 3] = df[meta_cols["snr_dB"]].values.astype(np.float32)

    # Noise variance
    if meta_cols["nVar"]:
        features[:, 4] = df[meta_cols["nVar"]].values.astype(np.float32)

    return features


def extract_modcod_features(df, meta_cols):
    """Extract MODCOD context: [modulation_order, code_rate].

    Parameters
    ----------
    df : DataFrame
    meta_cols : dict

    Returns
    -------
    modcod : (N, 2) float32  – [M_k, R_k]
    """
    N = len(df)
    modcod = np.zeros((N, 2), dtype=np.float32)

    if meta_cols["modcod"]:
        indices = df[meta_cols["modcod"]].values
        for i in range(N):
            m, r = modcod_index_to_features(indices[i])
            modcod[i, 0] = m
            modcod[i, 1] = r
    else:
        print("  WARNING: modcod column not found, using default QPSK 1/2")
        modcod[:, 0] = 4.0
        modcod[:, 1] = 0.5

    return modcod


# ============================================================================
# Sliding Window Builder
# ============================================================================
def build_sliding_windows_stage2(features, modcod, snr_db, history_len, horizon):
    """Build (X, Y) for stage-2 model from pre-extracted features.

    X : (n_samples, T_h, d_in)  where d_in = d_s + 2 = 7
    Y : (n_samples, H)          future SNR targets
    """
    combined = np.concatenate([features, modcod], axis=-1)  # (N, 7)
    N = len(snr_db)
    n_samples = N - history_len - horizon + 1

    if n_samples <= 0:
        print(f"\n  ERROR: Not enough frames ({N}) for T_h={history_len} + "
              f"H={horizon} = {history_len + horizon}")
        sys.exit(1)

    d_in = combined.shape[-1]
    X = np.zeros((n_samples, history_len, d_in), dtype=np.float32)
    Y = np.zeros((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        X[i] = combined[i: i + history_len]
        Y[i] = snr_db[i + history_len: i + history_len + horizon]

    return X, Y


def build_sliding_windows_full(blocks, masks, modcod, snr_db,
                                history_len, horizon):
    """Build sliding windows for the full two-stage model.

    Returns
    -------
    pilot_blocks_w : (n_samples, T_h, B, N_max, 2)
    pilot_masks_w  : (n_samples, T_h, B, N_max)
    modcod_w       : (n_samples, T_h, 2)
    Y              : (n_samples, H)
    """
    N = len(snr_db)
    n_samples = N - history_len - horizon + 1

    if n_samples <= 0:
        print(f"\n  ERROR: Not enough frames ({N}) for T_h={history_len} + "
              f"H={horizon} = {history_len + horizon}")
        sys.exit(1)

    B, N_max, C = blocks.shape[1], blocks.shape[2], blocks.shape[3]

    # Use memory-efficient approach: save indices and generate on-the-fly
    # For reasonably sized datasets, pre-build all windows
    print(f"  Allocating full-model windows: {n_samples} samples x "
          f"T_h={history_len} ...")

    # Estimate memory
    mem_gb = (n_samples * history_len * B * N_max * C * 4) / 1e9
    print(f"  Estimated pilot_blocks memory: {mem_gb:.2f} GB")

    if mem_gb > 8.0:
        print(f"  WARNING: Dataset would require {mem_gb:.1f} GB RAM!")
        print("  Consider reducing --history_len or number of frames.")
        print("  Saving window indices instead (use generator at train time).")
        # Save just the indices and raw data
        Y = np.zeros((n_samples, horizon), dtype=np.float32)
        for i in range(n_samples):
            Y[i] = snr_db[i + history_len: i + history_len + horizon]
        return None, None, None, Y  # Signal to use generator

    pilot_blocks_w = np.zeros(
        (n_samples, history_len, B, N_max, C), dtype=np.float32)
    pilot_masks_w = np.zeros(
        (n_samples, history_len, B, N_max), dtype=np.float32)
    modcod_w = np.zeros(
        (n_samples, history_len, 2), dtype=np.float32)
    Y = np.zeros((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        pilot_blocks_w[i] = blocks[i: i + history_len]
        pilot_masks_w[i] = masks[i: i + history_len]
        modcod_w[i] = modcod[i: i + history_len]
        Y[i] = snr_db[i + history_len: i + history_len + horizon]

    return pilot_blocks_w, pilot_masks_w, modcod_w, Y


# ============================================================================
# Normalization
# ============================================================================
def compute_norm_stats(X_train):
    """Compute per-feature mean/std from training data."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    return {
        "mean": flat.mean(axis=0).astype(np.float32),
        "std": flat.std(axis=0).astype(np.float32) + 1e-8,
    }


def normalize(X, stats):
    """Z-score normalization."""
    return ((X - stats["mean"]) / stats["std"]).astype(np.float32)


# ============================================================================
# Main Pipeline
# ============================================================================
def preprocess(args):
    print("=" * 70)
    print(f"  DVB-S2 Channel Dataset Preprocessor  (mode={args.mode})")
    print("=" * 70)

    # --- Step 1: Load CSV -------------------------------------------------
    print("\n[1/6] Loading CSV ...")
    df = load_csv(args.csv_path, delimiter=args.delimiter)

    # --- Step 2: Detect columns -------------------------------------------
    print("\n[2/6] Detecting columns ...")
    re_cols, im_cols, n_pilots = detect_pilot_columns(df)

    print("\n  Metadata columns:")
    meta_cols = detect_metadata_columns(df)

    # --- Step 3: Extract channel features ---------------------------------
    print("\n[3/6] Extracting channel features ...")
    features = extract_channel_features(df, meta_cols)      # (N, 5)
    modcod = extract_modcod_features(df, meta_cols)          # (N, 2)

    # Get SNR target vector
    snr_col = meta_cols["snr_dB"]
    if snr_col:
        snr_db = df[snr_col].values.astype(np.float32)
    else:
        print("  ERROR: snr_dB column is required as prediction target.")
        sys.exit(1)

    # Print feature statistics
    feat_names = ["channel_gain", "attenuation_dB", "phase_rad",
                  "snr_dB", "noise_var", "modcod_M", "modcod_R"]
    all_feats = np.concatenate([features, modcod], axis=-1)
    print(f"\n  {'Feature':<20s} {'Min':>10s} {'Max':>10s} "
          f"{'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(feat_names):
        col = all_feats[:, i]
        print(f"  {name:<20s} {col.min():>10.4f} {col.max():>10.4f} "
              f"{col.mean():>10.4f} {col.std():>10.4f}")

    # --- Step 4: Build sliding windows ------------------------------------
    T_h = args.history_len
    H = args.horizon
    print(f"\n[4/6] Building sliding windows (T_h={T_h}, H={H}) ...")

    # Stage-2 windows (always built – lightweight)
    X_s2, Y_s2 = build_sliding_windows_stage2(
        features, modcod, snr_db, T_h, H)
    print(f"  Stage-2: X={X_s2.shape}, Y={Y_s2.shape}")

    # Full-model pilot blocks (optional)
    if args.mode == "full":
        print("\n  Building pilot blocks ...")
        blocks, masks, n_max = build_pilot_blocks(
            df, re_cols, im_cols, n_pilots, args.num_blocks, args.n_max)
        pb_w, pm_w, mc_w, Y_full = build_sliding_windows_full(
            blocks, masks, modcod, snr_db, T_h, H)
    else:
        n_max = args.n_max

    # --- Step 5: Train / Val / Test split ---------------------------------
    print(f"\n[5/6] Splitting dataset "
          f"({args.train_frac}/{args.val_frac}/{args.test_frac}) ...")
    n = X_s2.shape[0]
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)

    # Stage-2 splits (temporal, no shuffle)
    X_train, Y_train = X_s2[:n_train], Y_s2[:n_train]
    X_val, Y_val = X_s2[n_train:n_train + n_val], Y_s2[n_train:n_train + n_val]
    X_test, Y_test = X_s2[n_train + n_val:], Y_s2[n_train + n_val:]

    print(f"  Train : {X_train.shape[0]:>6,} samples")
    print(f"  Val   : {X_val.shape[0]:>6,} samples")
    print(f"  Test  : {X_test.shape[0]:>6,} samples")

    # Normalize stage-2 features
    print("\n  Normalizing features (z-score from training set) ...")
    norm_stats = compute_norm_stats(X_train)
    X_train = normalize(X_train, norm_stats)
    X_val = normalize(X_val, norm_stats)
    X_test = normalize(X_test, norm_stats)

    # --- Step 6: Save to disk ---------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[6/6] Saving to: {output_dir.resolve()}")

    # Stage-2 data
    np.savez_compressed(output_dir / "stage2_train.npz", X=X_train, Y=Y_train)
    np.savez_compressed(output_dir / "stage2_val.npz", X=X_val, Y=Y_val)
    np.savez_compressed(output_dir / "stage2_test.npz", X=X_test, Y=Y_test)
    np.savez(output_dir / "norm_stats.npz", **norm_stats)
    print("  Saved: stage2_train.npz, stage2_val.npz, stage2_test.npz")

    # Full-model data
    if args.mode == "full" and pb_w is not None:
        np.savez_compressed(
            output_dir / "full_train.npz",
            pilot_blocks=pb_w[:n_train],
            pilot_masks=pm_w[:n_train],
            modcod=mc_w[:n_train],
            Y=Y_full[:n_train],
        )
        np.savez_compressed(
            output_dir / "full_val.npz",
            pilot_blocks=pb_w[n_train:n_train + n_val],
            pilot_masks=pm_w[n_train:n_train + n_val],
            modcod=mc_w[n_train:n_train + n_val],
            Y=Y_full[n_train:n_train + n_val],
        )
        np.savez_compressed(
            output_dir / "full_test.npz",
            pilot_blocks=pb_w[n_train + n_val:],
            pilot_masks=pm_w[n_train + n_val:],
            modcod=mc_w[n_train + n_val:],
            Y=Y_full[n_train + n_val:],
        )
        print("  Saved: full_train.npz, full_val.npz, full_test.npz")
    elif args.mode == "full":
        # Too large for memory – save raw arrays + indices
        np.save(output_dir / "pilot_blocks_raw.npy", blocks)
        np.save(output_dir / "pilot_masks_raw.npy", masks)
        np.save(output_dir / "modcod_raw.npy", modcod)
        np.save(output_dir / "snr_db_raw.npy", snr_db)
        print("  Saved raw arrays (use data generator for training)")

    # Config
    config = {
        "csv_path": str(args.csv_path),
        "mode": args.mode,
        "n_pilots_per_frame": n_pilots,
        "num_blocks": args.num_blocks,
        "n_max": int(n_max),
        "c_pilot": 2,
        "d_s": 5,
        "d_in": 7,
        "history_len": T_h,
        "prediction_horizon": H,
        "total_frames": int(df.shape[0]),
        "total_samples": int(n),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "feature_order": feat_names,
        "norm_mean": norm_stats["mean"].tolist(),
        "norm_std": norm_stats["std"].tolist(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    total_bytes = sum(f.stat().st_size for f in output_dir.iterdir()
                      if f.is_file())
    print(f"\n  Files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            unit = "KB" if size < 1e6 else "MB"
            val = size / 1024 if size < 1e6 else size / 1e6
            print(f"    {f.name:<30s}  {val:>8.1f} {unit}")

    print(f"\n  Total: {total_bytes / 1e6:.1f} MB")
    print("=" * 70)
    print(f"  Done!  Next: python train.py --data_dir {args.output_dir}")
    print("=" * 70)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess DVB-S2 MATLAB CSV for model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH,
                   help="Path to the MATLAB-exported CSV file.")
    p.add_argument("--output_dir", type=str, default="data",
                   help="Output directory for preprocessed .npz files.")
    p.add_argument("--mode", choices=["stage2", "full"], default="stage2",
                   help="'stage2': pre-extracted features only; "
                        "'full': also save pilot blocks for CNN.")
    p.add_argument("--delimiter", type=str, default=",",
                   help="CSV column delimiter.")
    p.add_argument("--num_blocks", type=int, default=16,
                   help="B: number of pilot blocks per frame.")
    p.add_argument("--n_max", type=int, default=50,
                   help="N_max: max pilots per block (auto-increased if needed).")
    p.add_argument("--history_len", type=int, default=200,
                   help="T_h: input history window length (frames).")
    p.add_argument("--horizon", type=int, default=50,
                   help="H: prediction horizon (future frames).")
    p.add_argument("--train_frac", type=float, default=0.70,
                   help="Fraction for training.")
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="Fraction for validation.")
    p.add_argument("--test_frac", type=float, default=0.15,
                   help="Fraction for testing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args)
