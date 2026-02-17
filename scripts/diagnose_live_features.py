#!/usr/bin/env python3
"""
Diagnostic script to compare live MT5 feature values against training data ranges.

This helps identify if live features are within expected distributions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple

from src.live.bridge_constants import MODEL_FEATURE_COLS, MARKET_FEATURE_COLS


def load_training_stats() -> Tuple[pd.DataFrame, Dict]:
    """Load training data and compute feature statistics."""
    processed = PROJECT_ROOT / "data" / "processed"

    # Load normalized parquet files
    df_5m = pd.read_parquet(processed / "features_5m_normalized.parquet")

    # Load normalizer to get raw statistics
    normalizer_path = PROJECT_ROOT / "models" / "analyst" / "normalizer_5m.pkl"
    with open(normalizer_path, 'rb') as f:
        normalizer = pickle.load(f)

    return df_5m, normalizer


def compute_feature_ranges(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Compute min, max, mean, std for each feature column."""
    stats = []
    for col in cols:
        if col in df.columns:
            data = df[col].dropna()
            stats.append({
                'feature': col,
                'min': float(data.min()),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'p5': float(data.quantile(0.05)),
                'p95': float(data.quantile(0.95)),
            })
        else:
            stats.append({
                'feature': col,
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'p5': np.nan,
                'p95': np.nan,
            })
    return pd.DataFrame(stats)


def print_expected_ranges():
    """Print expected feature ranges from training data."""
    print("=" * 80)
    print("FEATURE DIAGNOSTIC: Expected Ranges from Training Data")
    print("=" * 80)

    df_5m, normalizer = load_training_stats()

    print("\n### Normalizer Statistics (Raw Features Before Normalization) ###")
    print("-" * 80)
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    for col in normalizer.get('feature_cols', []):
        mean = normalizer['means'].get(col, np.nan)
        std = normalizer['stds'].get(col, np.nan)
        print(f"{col:<20} {mean:>12.6f} {std:>12.6f}")

    print("\n### MODEL_FEATURE_COLS (Analyst Input) - Normalized Values ###")
    print("-" * 80)
    stats = compute_feature_ranges(df_5m, MODEL_FEATURE_COLS)
    print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'P5':>10} {'P95':>10}")
    print("-" * 80)
    for _, row in stats.iterrows():
        print(f"{row['feature']:<20} {row['min']:>10.4f} {row['max']:>10.4f} "
              f"{row['mean']:>10.4f} {row['std']:>10.4f} {row['p5']:>10.4f} {row['p95']:>10.4f}")

    print("\n### MARKET_FEATURE_COLS (PPO Agent Input) - Normalized Values ###")
    print("-" * 80)
    stats = compute_feature_ranges(df_5m, MARKET_FEATURE_COLS)
    print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'P5':>10} {'P95':>10}")
    print("-" * 80)
    for _, row in stats.iterrows():
        print(f"{row['feature']:<20} {row['min']:>10.4f} {row['max']:>10.4f} "
              f"{row['mean']:>10.4f} {row['std']:>10.4f} {row['p5']:>10.4f} {row['p95']:>10.4f}")

    print("\n### Analyst Probability Distribution in Training Data ###")
    print("-" * 80)
    # Load analyst cache to see probability distribution during training
    cache_path = PROJECT_ROOT / "data" / "processed" / "analyst_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        probs = cache['probs']
        p_down = probs[:, 0]
        p_up = probs[:, 1]

        print(f"p_down: min={p_down.min():.4f} max={p_down.max():.4f} mean={p_down.mean():.4f} std={p_down.std():.4f}")
        print(f"p_up:   min={p_up.min():.4f} max={p_up.max():.4f} mean={p_up.mean():.4f} std={p_up.std():.4f}")

        # Distribution of conviction levels
        high_conviction = (np.abs(p_up - 0.5) > 0.2).sum()
        medium_conviction = ((np.abs(p_up - 0.5) > 0.1) & (np.abs(p_up - 0.5) <= 0.2)).sum()
        low_conviction = (np.abs(p_up - 0.5) <= 0.1).sum()
        total = len(p_up)

        print(f"\nConviction Distribution:")
        print(f"  High (|p_up - 0.5| > 0.2):   {high_conviction:>7,} ({100*high_conviction/total:.1f}%)")
        print(f"  Medium (0.1 < |p_up - 0.5| <= 0.2): {medium_conviction:>7,} ({100*medium_conviction/total:.1f}%)")
        print(f"  Low (|p_up - 0.5| <= 0.1):   {low_conviction:>7,} ({100*low_conviction/total:.1f}%)")

        # Bins for histogram
        print(f"\np_up Distribution Histogram:")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(p_up, bins=bins)
        for i in range(len(bins) - 1):
            bar = '█' * int(50 * hist[i] / hist.max()) if hist.max() > 0 else ''
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>7,} {bar}")
    else:
        print("Analyst cache not found - run pipeline to generate")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
If your LIVE features are outside these ranges, that explains the ~50% probabilities.

Common issues:
1. RAW_FEATURES values should be close to normalizer means (before normalization)
2. After normalization, most values should be in [-3, +3] range (z-scores)
3. Session flags should be 0 or 1 (not normalized)
4. NaN count should always be 0

Your live output shows:
  ANALYST_OUTPUT | context: min=-0.853 max=3.038 mean=0.002 | probs: [0.528, 0.472]

This means:
- Context vector looks reasonable (has variance, not all zeros)
- Probabilities near 50% = analyst is uncertain

To see MORE diagnostic info, enable DEBUG logging:
  export LOGLEVEL=DEBUG
  python scripts/run_mt5_bridge.py ...

Or add to the bridge command:
  python -c "import logging; logging.basicConfig(level=logging.DEBUG)" && python scripts/run_mt5_bridge.py ...
""")


def analyze_recent_decisions(log_file: str = None):
    """Parse recent bridge logs to analyze feature patterns."""
    if log_file is None:
        log_file = PROJECT_ROOT / "bridge_debug.log"

    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return

    print("\n### Analyzing Recent Decisions from Log ###")
    print("-" * 80)

    decisions = []
    with open(log_file, 'r') as f:
        for line in f:
            if "ANALYST_OUTPUT" in line:
                # Parse context stats
                try:
                    parts = line.split("|")
                    for p in parts:
                        if "context:" in p:
                            # Extract min/max/mean
                            pass
                        if "probs:" in p:
                            # Extract probabilities
                            pass
                except:
                    pass
            if "Decision @" in line:
                decisions.append(line.strip())

    print(f"Found {len(decisions)} decisions in log")
    if decisions:
        print("\nLast 5 decisions:")
        for d in decisions[-5:]:
            print(f"  {d}")


if __name__ == "__main__":
    print_expected_ranges()

    # Check for log file
    import sys
    if len(sys.argv) > 1:
        analyze_recent_decisions(sys.argv[1])
