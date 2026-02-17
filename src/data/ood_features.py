"""
v37 Training-Anchored OOD Features.

Unlike v36 rolling-window features that adapt to new data, these features
compare against FIXED training statistics that never change. This enables
true out-of-distribution detection during OOS/live trading.

Key difference from v36:
- v36: compute_volatility_regime() uses rolling(100).mean() → ADAPTS to new data
- v37: volatility_vs_training compares against FIXED training_vol_mean → NEVER adapts

Usage:
    # During training: compute and save baseline
    baseline = TrainingBaseline.from_training_data(train_df)
    baseline.save(model_dir / 'training_baseline.json')

    # During OOS/live: load baseline and compute features
    baseline = TrainingBaseline.load(model_dir / 'training_baseline.json')
    df = compute_training_anchored_ood_features(df, baseline)
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingBaseline:
    """
    Fixed statistics from training data - NEVER adapts to new data.

    These values are computed once during training and saved.
    During OOS/live, they're loaded and used as fixed reference points.
    """
    # Returns statistics
    returns_mean: float
    returns_std: float
    returns_skewness: float
    returns_kurtosis: float

    # Volatility statistics
    volatility_mean: float  # Rolling volatility mean
    volatility_std: float   # Rolling volatility std

    # ATR statistics (if available)
    atr_mean: float
    atr_std: float

    # Price level statistics
    price_mean: float
    price_std: float

    # Range statistics
    range_mean: float  # High-Low range
    range_std: float

    # Metadata
    n_samples: int
    start_date: str
    end_date: str

    @classmethod
    def from_training_data(cls, df: pd.DataFrame) -> 'TrainingBaseline':
        """
        Compute baseline statistics from training data.

        Args:
            df: Training DataFrame with columns: open, high, low, close
                Optionally: atr (if pre-computed)

        Returns:
            TrainingBaseline with fixed statistics
        """
        # Compute returns
        returns = df['close'].pct_change().dropna()

        # Returns statistics
        returns_mean = float(returns.mean())
        returns_std = float(returns.std())
        returns_skewness = float(returns.skew())
        returns_kurtosis = float(returns.kurtosis())

        # Rolling volatility (20-period std of returns, annualized for consistency)
        rolling_vol = returns.rolling(20).std().dropna()
        volatility_mean = float(rolling_vol.mean())
        volatility_std = float(rolling_vol.std())

        # ATR if available, else compute from HLC
        if 'atr' in df.columns:
            atr = df['atr'].dropna()
        else:
            # Compute True Range
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().dropna()

        atr_mean = float(atr.mean())
        atr_std = float(atr.std())

        # Price statistics
        price_mean = float(df['close'].mean())
        price_std = float(df['close'].std())

        # Range statistics (High-Low)
        ranges = df['high'] - df['low']
        range_mean = float(ranges.mean())
        range_std = float(ranges.std())

        # Metadata
        n_samples = len(df)

        # Handle timestamp extraction
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = str(df.index[0])
            end_date = str(df.index[-1])
        elif 'timestamp' in df.columns:
            start_date = str(df['timestamp'].iloc[0])
            end_date = str(df['timestamp'].iloc[-1])
        else:
            start_date = "unknown"
            end_date = "unknown"

        baseline = cls(
            returns_mean=returns_mean,
            returns_std=returns_std,
            returns_skewness=returns_skewness,
            returns_kurtosis=returns_kurtosis,
            volatility_mean=volatility_mean,
            volatility_std=volatility_std,
            atr_mean=atr_mean,
            atr_std=atr_std,
            price_mean=price_mean,
            price_std=price_std,
            range_mean=range_mean,
            range_std=range_std,
            n_samples=n_samples,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"TrainingBaseline computed from {n_samples} samples")
        logger.info(f"  Returns: mean={returns_mean:.6f}, std={returns_std:.6f}, skew={returns_skewness:.3f}")
        logger.info(f"  Volatility: mean={volatility_mean:.6f}, std={volatility_std:.6f}")
        logger.info(f"  ATR: mean={atr_mean:.2f}, std={atr_std:.2f}")
        logger.info(f"  Price: mean={price_mean:.2f}, std={price_std:.2f}")

        return baseline

    def save(self, path: Path) -> None:
        """Save baseline to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        logger.info(f"TrainingBaseline saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'TrainingBaseline':
        """Load baseline from JSON file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"TrainingBaseline not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        baseline = cls(**data)
        logger.info(f"TrainingBaseline loaded from {path}")
        logger.info(f"  Training period: {baseline.start_date} to {baseline.end_date}")
        logger.info(f"  Samples: {baseline.n_samples}")

        return baseline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def compute_training_anchored_ood_features(
    df: pd.DataFrame,
    baseline: TrainingBaseline,
    window: int = 20
) -> pd.DataFrame:
    """
    Compute OOD features by comparing current market against FIXED training baseline.

    Unlike v36 rolling features, these compare against immutable training statistics,
    enabling true out-of-distribution detection that doesn't adapt to new data.

    Args:
        df: DataFrame with OHLC data
        baseline: TrainingBaseline with fixed training statistics
        window: Rolling window for computing current statistics

    Returns:
        DataFrame with new columns:
        - volatility_vs_training: Z-score of current vol vs training vol
        - returns_skew_shift: Absolute difference from training skewness
        - atr_vs_training: Z-score of current ATR vs training ATR
        - ood_score: Composite 0-1 score (higher = more OOD)
    """
    df = df.copy()

    # Compute current returns
    returns = df['close'].pct_change()

    # --- Volatility vs Training ---
    # Current rolling volatility
    current_vol = returns.rolling(window).std()
    # Z-score vs FIXED training statistics
    vol_z = (current_vol - baseline.volatility_mean) / max(baseline.volatility_std, 1e-8)
    df['volatility_vs_training'] = vol_z.fillna(0.0).astype(np.float32)

    # --- Returns Skewness Shift ---
    # Rolling skewness of recent returns
    current_skew = returns.rolling(window * 5).skew()  # Need more samples for skewness
    # Forward-fill NaNs from first valid, then backfill remaining (avoids artificial 0-skewness)
    current_skew = current_skew.ffill().bfill().fillna(0.0)
    # Absolute difference from FIXED training skewness
    skew_shift = (current_skew - baseline.returns_skewness).abs()
    df['returns_skew_shift'] = skew_shift.astype(np.float32)

    # --- ATR vs Training ---
    # Compute current ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    current_atr = true_range.rolling(14).mean()
    # Z-score vs FIXED training ATR
    atr_z = (current_atr - baseline.atr_mean) / max(baseline.atr_std, 1e-8)
    df['atr_vs_training'] = atr_z.fillna(0.0).astype(np.float32)

    # --- Composite OOD Score ---
    # Combine multiple signals into 0-1 score
    # Higher score = more out-of-distribution

    # Volatility component: abs z-score, normalized to 0-1
    vol_component = np.clip(np.abs(df['volatility_vs_training']) / 3.0, 0, 1)

    # Skewness component: shift normalized to 0-1 (skew shift > 2 is severe)
    skew_component = np.clip(df['returns_skew_shift'] / 2.0, 0, 1)

    # ATR component: abs z-score, normalized to 0-1
    atr_component = np.clip(np.abs(df['atr_vs_training']) / 3.0, 0, 1)

    # Weighted average (volatility and skewness are most important based on our analysis)
    df['ood_score'] = (
        0.40 * vol_component +
        0.35 * skew_component +
        0.25 * atr_component
    ).astype(np.float32)

    # Log summary statistics
    mean_ood = df['ood_score'].mean()
    max_ood = df['ood_score'].max()
    pct_high_ood = (df['ood_score'] > 0.5).mean() * 100

    logger.info(f"v37 OOD Features computed:")
    logger.info(f"  Mean OOD score: {mean_ood:.3f}")
    logger.info(f"  Max OOD score: {max_ood:.3f}")
    logger.info(f"  % High OOD (>0.5): {pct_high_ood:.1f}%")

    if mean_ood > 0.4:
        logger.warning(f"HIGH OOD DETECTED - Mean score {mean_ood:.3f} > 0.4")
        logger.warning("Model may significantly underperform on this data")

    return df


def get_ood_summary(df: pd.DataFrame, baseline: TrainingBaseline) -> Dict[str, Any]:
    """
    Get detailed OOD summary for diagnostics.

    Args:
        df: DataFrame with OOD features computed
        baseline: TrainingBaseline for reference

    Returns:
        Dictionary with OOD summary statistics
    """
    if 'ood_score' not in df.columns:
        raise ValueError("OOD features not computed. Call compute_training_anchored_ood_features first.")

    returns = df['close'].pct_change().dropna()

    # Current data statistics
    current_stats = {
        'returns_mean': float(returns.mean()),
        'returns_std': float(returns.std()),
        'returns_skewness': float(returns.skew()),
        'price_mean': float(df['close'].mean()),
    }

    # Shifts from training
    shifts = {
        'returns_mean_shift': current_stats['returns_mean'] - baseline.returns_mean,
        'returns_std_ratio': current_stats['returns_std'] / max(baseline.returns_std, 1e-8),
        'skewness_shift': current_stats['returns_skewness'] - baseline.returns_skewness,
        'price_level_change_pct': (current_stats['price_mean'] - baseline.price_mean) / baseline.price_mean * 100,
    }

    # OOD score statistics
    ood_stats = {
        'ood_score_mean': float(df['ood_score'].mean()),
        'ood_score_std': float(df['ood_score'].std()),
        'ood_score_max': float(df['ood_score'].max()),
        'pct_high_ood': float((df['ood_score'] > 0.5).mean() * 100),
        'pct_severe_ood': float((df['ood_score'] > 0.7).mean() * 100),
    }

    # Feature-level statistics
    feature_stats = {}
    for col in ['volatility_vs_training', 'returns_skew_shift', 'atr_vs_training']:
        if col in df.columns:
            feature_stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
            }

    return {
        'training_baseline': baseline.to_dict(),
        'current_stats': current_stats,
        'shifts': shifts,
        'ood_stats': ood_stats,
        'feature_stats': feature_stats,
    }
