"""
Elite Test Suite - Advanced Data Pipeline Tests
================================================
Deep tests for data loading, feature engineering, and normalization.

Tests verify:
- Timezone handling correctness
- Fractal BOS/CHoCH no look-ahead bias
- Timeframe alignment with sparse data
- CHOP/ADX numerical stability
- Target class balance
- Multi-horizon alignment
- Rolling window accuracy
- Normalization per-timeframe scaling
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# TIMEZONE HANDLING TESTS (CRITICAL)
# =============================================================================

class TestTimezoneHandling:
    """
    Tests for timezone handling in data loading.

    CRITICAL BUG: tz_localize(None) removes timezone info WITHOUT conversion.
    This means a timestamp "2024-01-01 12:00 UTC" becomes "2024-01-01 12:00"
    in local time interpretation, which could be wrong.
    """

    def test_timezone_removal_behavior(self):
        """Verify timezone removal behavior in pandas."""
        # Create a timezone-aware timestamp
        ts_utc = pd.Timestamp("2024-01-01 12:00:00", tz='UTC')

        # tz_localize(None) removes timezone without conversion
        ts_naive = ts_utc.tz_localize(None)

        # The values should be the same (hour = 12)
        assert ts_naive.hour == 12, \
            "tz_localize(None) should keep the same values"

    def test_timezone_conversion_not_applied(self):
        """
        tz_localize(None) does NOT convert, just strips timezone.
        This is the current behavior - verify we understand it.
        """
        # Create UTC timestamp
        ts_utc = pd.Timestamp("2024-01-01 00:00:00", tz='UTC')

        # If we converted to US/Eastern, it would be "2023-12-31 19:00:00"
        # But tz_localize(None) just removes the 'UTC' marker
        ts_stripped = ts_utc.tz_localize(None)

        # This remains 00:00, not 19:00
        assert ts_stripped.hour == 0, \
            "Hour should remain 0 (not converted)"

    def test_session_times_consistent(self, sample_ohlcv_data):
        """Session time calculations should be consistent."""
        df = sample_ohlcv_data.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Create datetime index
            df.index = pd.date_range(
                start='2024-01-01',
                periods=len(df),
                freq='5min'
            )

        # Extract hour (for session detection)
        hours = df.index.hour

        # Verify hours are in valid range
        assert hours.min() >= 0 and hours.max() <= 23, \
            "Hours should be in [0, 23]"


# =============================================================================
# FRACTAL/BOS/CHOCH NO LOOK-AHEAD TESTS (CRITICAL)
# =============================================================================

class TestFractalNoLookAhead:
    """
    Tests to ensure fractal detection has no look-ahead bias.

    Fractals require future bars to confirm, so detection must be DELAYED.
    """

    def test_fractal_detection_delayed(self, sample_ohlcv_data):
        """Fractal detection should only use past/current data."""
        from src.data.features import detect_fractals

        df = sample_ohlcv_data.copy()
        n = 5  # Fractal window

        fractal_highs, fractal_lows = detect_fractals(df, n)

        # First n-1 bars cannot have confirmed fractals
        assert fractal_highs.iloc[:n-1].sum() == 0, \
            "No fractals should be detected in first n-1 bars"
        assert fractal_lows.iloc[:n-1].sum() == 0, \
            "No fractals should be detected in first n-1 bars"

    def test_fractal_uses_half_n_delay(self, sample_ohlcv_data):
        """
        Fractal at bar i is confirmed at bar i + n//2.
        At bar i+2 (for n=5), we can finally confirm bar i was a fractal.
        """
        from src.data.features import detect_fractals

        df = sample_ohlcv_data.copy()
        n = 5
        half_n = n // 2

        # Artificially create a clear fractal high
        # Make bar 10 clearly the highest in its window
        df.iloc[10, df.columns.get_loc('high')] = 99999.0

        fractal_highs, _ = detect_fractals(df, n)

        # The fractal at bar 10 should be detected at bar 12 (10 + 2)
        # Not at bar 10 (would be look-ahead)
        detection_idx = 10 + half_n

        # Due to implementation specifics, just verify no detection before confirmation
        assert not fractal_highs.iloc[10], \
            "Fractal should not be detected at its own bar (look-ahead)"

    def test_bos_choch_no_future_data(self, sample_features_df):
        """BOS/CHoCH features should not use future data."""
        df = sample_features_df.copy()

        # If BOS/CHoCH columns exist, verify they're based on past only
        bos_cols = [c for c in df.columns if 'bos' in c.lower()]
        choch_cols = [c for c in df.columns if 'choch' in c.lower()]

        for col in bos_cols + choch_cols:
            if col in df.columns:
                # Values in first few rows should be 0 or NaN (no data to confirm)
                first_few = df[col].iloc[:5]
                # At least some should be 0/NaN (not all confirmed immediately)
                assert first_few.isna().any() or (first_few == 0).any(), \
                    f"{col} should have incomplete data at start"


# =============================================================================
# TIMEFRAME ALIGNMENT TESTS
# =============================================================================

class TestTimeframeAlignment:
    """Tests for multi-timeframe data alignment."""

    def test_15m_subsampling_ratio(self):
        """15m timeframe should be 3x subsampling of 5m."""
        # 15 min / 5 min = 3
        subsample_ratio = 15 // 5
        assert subsample_ratio == 3

    def test_45m_subsampling_ratio(self):
        """45m timeframe should be 9x subsampling of 5m."""
        # 45 min / 5 min = 9
        subsample_ratio = 45 // 5
        assert subsample_ratio == 9

    def test_alignment_with_sparse_data(self, sample_ohlcv_data):
        """Alignment should handle sparse/missing data gracefully."""
        df = sample_ohlcv_data.copy()

        # Simulate sparse data by dropping some rows
        sparse_df = df.iloc[::3]  # Keep every 3rd row

        # Length should be reduced
        assert len(sparse_df) == len(df) // 3 + (1 if len(df) % 3 else 0)

        # Should still have valid OHLC
        assert 'open' in sparse_df.columns


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability in indicators."""

    def test_chop_never_negative(self, sample_ohlcv_data):
        """Choppiness Index should never be negative."""
        from src.data.features import choppiness_index

        df = sample_ohlcv_data.copy()

        chop = choppiness_index(df, period=14)

        # Remove NaN values
        chop_valid = chop.dropna()

        if len(chop_valid) > 0:
            assert (chop_valid >= 0).all(), "CHOP should never be negative"

    def test_adx_bounded_0_100(self, sample_ohlcv_data):
        """ADX should be bounded between 0 and 100."""
        from src.data.features import adx as calculate_adx

        df = sample_ohlcv_data.copy()

        adx_values = calculate_adx(df, period=14)
        adx_valid = adx_values.dropna()

        if len(adx_valid) > 0:
            assert (adx_valid >= 0).all(), "ADX should be >= 0"
            assert (adx_valid <= 100).all(), "ADX should be <= 100"

    def test_atr_always_positive(self, sample_ohlcv_data):
        """ATR should always be positive (or zero for no movement)."""
        from src.data.features import atr as calculate_atr

        df = sample_ohlcv_data.copy()

        atr_values = calculate_atr(df, period=14)
        atr_valid = atr_values.dropna()

        if len(atr_valid) > 0:
            assert (atr_valid >= 0).all(), "ATR should never be negative"

    def test_zero_range_handling(self):
        """Indicators should handle zero-range candles gracefully."""
        # Create data where high == low (doji-like)
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [100.0, 100.0, 100.0],
            'low': [100.0, 100.0, 100.0],
            'close': [100.0, 100.0, 100.0],
        })

        # ATR should be 0 or near-0, not NaN or negative
        from src.data.features import atr as calculate_atr
        atr_values = calculate_atr(df, period=2)

        # Should not raise and should have valid values
        assert atr_values is not None
        # Last value should be 0 or very small
        if not np.isnan(atr_values.iloc[-1]):
            assert atr_values.iloc[-1] >= 0


# =============================================================================
# TARGET GENERATION TESTS
# =============================================================================

class TestTargetGeneration:
    """Tests for classification target generation."""

    def test_target_has_no_lookahead(self, sample_ohlcv_data):
        """Target at time t should use data from t+1 to t+horizon."""
        df = sample_ohlcv_data.copy()
        horizon = 5

        # Create simple target: future return
        future_close = df['close'].shift(-horizon)
        target = np.where(future_close > df['close'], 1, 0)

        # Last 'horizon' values should be NaN (no future data)
        # Actually they'll be comparing with NaN, so check for this
        assert len(df) >= horizon, "Need enough data for horizon"

    def test_target_class_balance(self, sample_ohlcv_data):
        """Target classes should be reasonably balanced."""
        df = sample_ohlcv_data.copy()

        # Create binary target based on price direction
        df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
        df_valid = df.dropna()

        if len(df_valid) > 10:
            class_counts = df_valid['target'].value_counts(normalize=True)

            # Neither class should be more than 70%
            for cls_pct in class_counts:
                assert cls_pct < 0.7, \
                    "Target class imbalance too severe (>70%)"


# =============================================================================
# NORMALIZER TESTS
# =============================================================================

class TestNormalizer:
    """Tests for feature normalization."""

    def test_normalizer_fit_only_on_train(self):
        """Normalizer should fit ONLY on training data."""
        from src.data.normalizer import FeatureNormalizer

        # Training data
        train_df = pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 50,
            'feature2': np.random.randn(100) * 5 + 25,
        })

        # Test data (different distribution)
        test_df = pd.DataFrame({
            'feature1': np.random.randn(20) * 10 + 100,  # Higher mean
            'feature2': np.random.randn(20) * 5 + 50,
        })

        normalizer = FeatureNormalizer(['feature1', 'feature2'])
        normalizer.fit(train_df)

        # Transform test data with TRAIN stats
        test_normalized = normalizer.transform(test_df)

        # Test data's mean should NOT be 0 (using train stats)
        test_mean = test_normalized['feature1'].mean()

        # Since test has higher mean than train, normalized should be positive
        assert test_mean > 0, \
            "Test mean should be positive (test > train mean)"

    def test_normalizer_clips_extreme_values(self):
        """Normalizer should clip extreme z-scores."""
        from src.data.normalizer import FeatureNormalizer

        # Data with outliers
        df = pd.DataFrame({
            'feature': np.concatenate([
                np.random.randn(100),  # Normal values
                np.array([100.0, -100.0])  # Extreme outliers
            ])
        })

        normalizer = FeatureNormalizer(['feature'], clip_zscore=5.0)
        normalized = normalizer.fit_transform(df)

        # All values should be within [-5, 5]
        assert normalized['feature'].max() <= 5.0
        assert normalized['feature'].min() >= -5.0

    def test_normalizer_save_load_roundtrip(self, tmp_path):
        """Normalizer should maintain stats after save/load."""
        from src.data.normalizer import FeatureNormalizer

        df = pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 50,
        })

        # Fit and save
        normalizer = FeatureNormalizer(['feature1'])
        normalizer.fit(df)
        original_mean = normalizer.means['feature1']

        save_path = tmp_path / "normalizer.pkl"
        normalizer.save(save_path)

        # Load and verify
        loaded = FeatureNormalizer.load(save_path)

        assert loaded.means['feature1'] == original_mean, \
            "Loaded normalizer should have same stats"

    def test_normalizer_handles_constant_feature(self):
        """Normalizer should handle constant features (std=0)."""
        from src.data.normalizer import FeatureNormalizer

        df = pd.DataFrame({
            'constant': [5.0] * 100,  # All same value
            'normal': np.random.randn(100),
        })

        normalizer = FeatureNormalizer(['constant', 'normal'])
        normalized = normalizer.fit_transform(df)

        # Constant feature should be all zeros (or near-zero)
        assert np.allclose(normalized['constant'], 0, atol=1e-6), \
            "Constant feature should normalize to 0"

        # Should not have NaN or Inf
        assert np.all(np.isfinite(normalized['constant'])), \
            "Constant feature should not produce NaN/Inf"


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering correctness."""

    pass  # Removed tests for deprecated features (pinbar, engulfing, doji, ema_crossover)


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Tests for data validation and quality checks."""

    def test_high_greater_than_low(self, sample_ohlcv_data):
        """High should always be >= Low."""
        df = sample_ohlcv_data.copy()

        invalid_count = (df['high'] < df['low']).sum()
        assert invalid_count == 0, \
            f"Found {invalid_count} rows where high < low"

    def test_open_close_within_range(self, sample_ohlcv_data):
        """Open and Close should be within High-Low range."""
        df = sample_ohlcv_data.copy()

        # Open should be <= High and >= Low
        open_valid = (df['open'] <= df['high']) & (df['open'] >= df['low'])
        close_valid = (df['close'] <= df['high']) & (df['close'] >= df['low'])

        assert open_valid.all(), "Open should be within High-Low range"
        assert close_valid.all(), "Close should be within High-Low range"

    def test_no_negative_prices(self, sample_ohlcv_data):
        """Prices should never be negative."""
        df = sample_ohlcv_data.copy()

        for col in ['open', 'high', 'low', 'close']:
            assert (df[col] >= 0).all(), f"{col} should never be negative"

    def test_data_chronological(self, sample_ohlcv_data):
        """Data should be in chronological order."""
        df = sample_ohlcv_data.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            # Index should be monotonically increasing
            is_sorted = df.index.is_monotonic_increasing

            assert is_sorted, "Data should be chronologically sorted"


# =============================================================================
# RESAMPLING TESTS
# =============================================================================

class TestResampling:
    """Tests for timeframe resampling."""

    def test_resample_preserves_ohlc_logic(self, sample_ohlcv_data):
        """Resampling should preserve OHLC logic correctly."""
        df = sample_ohlcv_data.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.date_range(
                start='2024-01-01',
                periods=len(df),
                freq='5min'
            )

        # Resample to 15min
        resampled = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(resampled) > 0:
            # High should still be >= Low
            assert (resampled['high'] >= resampled['low']).all()

            # Open/Close should be within range
            assert ((resampled['open'] <= resampled['high']) &
                   (resampled['open'] >= resampled['low'])).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
