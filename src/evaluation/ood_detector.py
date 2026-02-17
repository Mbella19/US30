"""
Out-of-Distribution (OOD) Detection for Runtime Adaptation.

v36 OOD Fix: Detects when current market conditions deviate from training
data distribution, enabling adaptive responses like position size reduction.

Key Features:
- Compares live/OOS statistics against training baseline
- Tracks rolling volatility z-score
- Computes regime distribution shift
- Returns confidence multiplier for position sizing
"""

import json
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OODMetrics:
    """Metrics from OOD detection analysis."""
    volatility_z: float          # Z-score of current volatility vs training
    regime_shift_score: float    # KL divergence of regime distribution (0-1)
    is_ood: bool                 # Whether current market is out-of-distribution
    confidence_multiplier: float # Position size multiplier (0.3-1.0)
    details: Dict[str, float]    # Additional diagnostic info


class OODDetector:
    """
    Detects when current market conditions are out-of-distribution
    relative to training data statistics.

    Usage:
        # Initialize with training stats
        detector = OODDetector.from_training_stats(stats_path)

        # Or initialize manually
        detector = OODDetector(
            train_vol_mean=0.00072,
            train_vol_std=0.00035,
            train_regime_dist=np.array([0.33, 0.34, 0.33])
        )

        # Update with each observation
        metrics = detector.update(current_vol=0.00045, current_regime=1)

        # Use confidence multiplier for position sizing
        position_size *= metrics.confidence_multiplier
    """

    def __init__(
        self,
        train_vol_mean: float,
        train_vol_std: float,
        train_regime_dist: Optional[np.ndarray] = None,
        train_feature_means: Optional[np.ndarray] = None,
        train_feature_stds: Optional[np.ndarray] = None,
        ood_threshold: float = 2.0,
        buffer_size: int = 100
    ):
        """
        Initialize OOD detector with training statistics.

        Args:
            train_vol_mean: Mean volatility from training data
            train_vol_std: Std dev of volatility from training data
            train_regime_dist: Distribution of regimes in training [bullish, ranging, bearish]
            train_feature_means: Mean of all market features from training
            train_feature_stds: Std dev of all market features from training
            ood_threshold: Z-score threshold for flagging OOD (default: 2.0)
            buffer_size: Rolling buffer size for runtime statistics
        """
        self.train_vol_mean = train_vol_mean
        self.train_vol_std = max(train_vol_std, 1e-8)  # Prevent division by zero
        self.train_regime_dist = train_regime_dist
        self.train_feature_means = train_feature_means
        self.train_feature_stds = train_feature_stds
        self.ood_threshold = ood_threshold
        self.buffer_size = buffer_size

        # Rolling buffers for runtime stats
        self.vol_buffer: list = []
        self.regime_buffer: list = []

        logger.info(f"OOD Detector initialized:")
        logger.info(f"  Training vol mean: {train_vol_mean:.6f}")
        logger.info(f"  Training vol std:  {train_vol_std:.6f}")
        logger.info(f"  OOD threshold:     {ood_threshold} std devs")

    @classmethod
    def from_training_stats(cls, stats_path: str) -> 'OODDetector':
        """
        Create OODDetector from saved training statistics JSON file.

        Args:
            stats_path: Path to training_stats.json file

        Returns:
            Initialized OODDetector
        """
        path = Path(stats_path)
        if not path.exists():
            raise FileNotFoundError(f"Training stats not found: {stats_path}")

        with open(path, 'r') as f:
            stats = json.load(f)

        # Extract volatility stats
        vol_stats = stats.get('volatility_stats', {})
        train_vol_mean = vol_stats.get('atr_mean', 0.0)
        train_vol_std = vol_stats.get('atr_std', 1.0)

        # Extract feature stats
        train_feature_means = np.array(stats.get('market_feat_mean', []))
        train_feature_stds = np.array(stats.get('market_feat_std', []))

        # Extract regime distribution
        train_regime_dist = None
        if 'regime_ratios' in stats:
            ratios = stats['regime_ratios']
            train_regime_dist = np.array([
                ratios.get('0', 0.33),  # Bullish
                ratios.get('1', 0.34),  # Ranging
                ratios.get('2', 0.33),  # Bearish
            ])

        return cls(
            train_vol_mean=train_vol_mean,
            train_vol_std=train_vol_std,
            train_regime_dist=train_regime_dist,
            train_feature_means=train_feature_means if len(train_feature_means) > 0 else None,
            train_feature_stds=train_feature_stds if len(train_feature_stds) > 0 else None,
        )

    def update(
        self,
        current_vol: float,
        current_regime: Optional[int] = None,
        current_features: Optional[np.ndarray] = None
    ) -> OODMetrics:
        """
        Update detector with new observation and return OOD metrics.

        Args:
            current_vol: Current volatility (e.g., ATR)
            current_regime: Current regime label (0=Bullish, 1=Ranging, 2=Bearish)
            current_features: Full market feature vector (optional, for detailed analysis)

        Returns:
            OODMetrics with detection results
        """
        # Update rolling buffers
        self.vol_buffer.append(current_vol)
        if len(self.vol_buffer) > self.buffer_size:
            self.vol_buffer.pop(0)

        if current_regime is not None:
            self.regime_buffer.append(current_regime)
            if len(self.regime_buffer) > self.buffer_size:
                self.regime_buffer.pop(0)

        # Compute volatility z-score relative to training
        vol_z = (current_vol - self.train_vol_mean) / self.train_vol_std

        # Compute regime distribution shift
        regime_shift = 0.0
        if self.train_regime_dist is not None and len(self.regime_buffer) >= 50:
            current_regime_dist = self._compute_regime_dist()
            regime_shift = self._kl_divergence(current_regime_dist, self.train_regime_dist)

        # Compute feature-wise shift if features provided
        feature_shift = 0.0
        if current_features is not None and self.train_feature_means is not None:
            feature_z = np.abs(current_features - self.train_feature_means) / (self.train_feature_stds + 1e-8)
            feature_shift = float(np.mean(np.clip(feature_z, 0, 5)))

        # Determine if OOD
        # OOD if: volatility is significantly different OR regime has shifted significantly
        is_ood = abs(vol_z) > self.ood_threshold or regime_shift > 0.5

        # Compute confidence multiplier (reduce position size when OOD)
        # Uses smooth scaling: 1.0 at z=0, 0.5 at z=threshold, 0.3 at z=2*threshold
        if is_ood:
            # Sigmoid-like decay
            z_mag = abs(vol_z)
            confidence_multiplier = max(0.3, 1.0 - (z_mag / (2 * self.ood_threshold)))

            # Also reduce for regime shift
            if regime_shift > 0.3:
                confidence_multiplier *= max(0.5, 1.0 - regime_shift)
        else:
            confidence_multiplier = 1.0

        # Collect diagnostic details
        details = {
            'current_vol': current_vol,
            'train_vol_mean': self.train_vol_mean,
            'train_vol_std': self.train_vol_std,
            'vol_z': vol_z,
            'regime_shift': regime_shift,
            'feature_shift': feature_shift,
            'buffer_vol_mean': float(np.mean(self.vol_buffer)) if self.vol_buffer else 0.0,
            'buffer_vol_std': float(np.std(self.vol_buffer)) if len(self.vol_buffer) > 1 else 0.0,
        }

        return OODMetrics(
            volatility_z=vol_z,
            regime_shift_score=regime_shift,
            is_ood=is_ood,
            confidence_multiplier=confidence_multiplier,
            details=details
        )

    def _compute_regime_dist(self) -> np.ndarray:
        """Compute regime distribution from rolling buffer."""
        if not self.regime_buffer:
            return np.array([0.33, 0.34, 0.33])

        counts = np.bincount(self.regime_buffer, minlength=3)
        dist = counts / len(self.regime_buffer)
        return dist

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence D(P || Q).

        Returns a value >= 0, where 0 means identical distributions.
        """
        # Add small epsilon to prevent log(0)
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)

        # Normalize to ensure valid distributions
        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    def reset(self):
        """Reset rolling buffers."""
        self.vol_buffer = []
        self.regime_buffer = []

    def get_summary(self) -> Dict:
        """Get summary of current OOD detection state."""
        return {
            'buffer_size': len(self.vol_buffer),
            'buffer_vol_mean': float(np.mean(self.vol_buffer)) if self.vol_buffer else None,
            'buffer_vol_std': float(np.std(self.vol_buffer)) if len(self.vol_buffer) > 1 else None,
            'buffer_vol_z': (float(np.mean(self.vol_buffer)) - self.train_vol_mean) / self.train_vol_std if self.vol_buffer else None,
            'train_vol_mean': self.train_vol_mean,
            'train_vol_std': self.train_vol_std,
        }


def analyze_distribution_shift(
    train_features: np.ndarray,
    test_features: np.ndarray,
    feature_names: Optional[list] = None
) -> Dict:
    """
    Analyze distribution shift between training and test feature sets.

    Args:
        train_features: Training data features (n_samples, n_features)
        test_features: Test data features (n_samples, n_features)
        feature_names: Optional list of feature names

    Returns:
        Dictionary with shift analysis results
    """
    from scipy import stats

    n_features = train_features.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    results = {
        'features': [],
        'summary': {}
    }

    significant_shifts = 0

    for i, name in enumerate(feature_names):
        train_col = train_features[:, i]
        test_col = test_features[:, i]

        # Basic statistics
        train_mean = float(train_col.mean())
        test_mean = float(test_col.mean())
        train_std = float(train_col.std())
        test_std = float(test_col.std())

        # Mean shift in std units
        mean_shift_z = (test_mean - train_mean) / (train_std + 1e-8)

        # Variance ratio
        var_ratio = (test_std ** 2) / (train_std ** 2 + 1e-8)

        # KS test for distribution difference
        ks_stat, ks_pvalue = stats.ks_2samp(train_col, test_col)

        # Flag significant shifts
        is_significant = ks_pvalue < 0.01 or abs(mean_shift_z) > 1.0
        if is_significant:
            significant_shifts += 1

        results['features'].append({
            'name': name,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'mean_shift_z': float(mean_shift_z),
            'train_std': train_std,
            'test_std': test_std,
            'var_ratio': float(var_ratio),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'significant_shift': is_significant
        })

    # Sort by KS statistic (largest shifts first)
    results['features'].sort(key=lambda x: x['ks_statistic'], reverse=True)

    results['summary'] = {
        'n_features': n_features,
        'n_significant_shifts': significant_shifts,
        'pct_significant': significant_shifts / n_features * 100,
        'top_shifted_features': [f['name'] for f in results['features'][:5]]
    }

    return results
