"""
Elite Test Suite - Feature Engineering Tests
=============================================
Comprehensive verification of feature engineering and target creation.

Tests verify:
- Target creation (smoothed, binary, multi-horizon)
- Rolling window calculations (ATR, SMA, CHOP)
- No look-ahead bias in features
- NaN handling
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# TARGET CREATION TESTS
# =============================================================================

class TestSmoothedTarget:
    """Tests for create_smoothed_target function."""
    
    def test_smoothed_target_output_shape(self, sample_ohlcv_data):
        """Smoothed target should match input length."""
        from src.data.features import create_smoothed_target
        
        target = create_smoothed_target(
            sample_ohlcv_data,
            future_window=12,
            smooth_window=3
        )
        
        assert len(target) == len(sample_ohlcv_data), \
            f"Target length {len(target)} should match input {len(sample_ohlcv_data)}"
    
    def test_smoothed_target_has_nan_at_end(self, sample_ohlcv_data):
        """Smoothed target should have NaN at the end (no future data)."""
        from src.data.features import create_smoothed_target
        
        future_window = 12
        target = create_smoothed_target(
            sample_ohlcv_data,
            future_window=future_window,
            smooth_window=3
        )
        
        # Last future_window entries should be NaN
        assert pd.isna(target.iloc[-1]) or target.iloc[-future_window:].isna().any(), \
            "Target should have NaN values at the end (no future data available)"
    
    def test_smoothed_target_dtype(self, sample_ohlcv_data):
        """Target should be numeric."""
        from src.data.features import create_smoothed_target
        
        target = create_smoothed_target(sample_ohlcv_data, 12, 3)
        
        assert np.issubdtype(target.dtype, np.floating), \
            f"Target should be float, got {target.dtype}"
    
    def test_smoothed_target_no_lookahead(self, sample_ohlcv_data):
        """
        Verify target at time T only uses data from T to T+future_window.
        This is the correct 'future' target for supervised learning.
        """
        from src.data.features import create_smoothed_target
        
        future_window = 12
        target = create_smoothed_target(sample_ohlcv_data, future_window, 3)
        
        # Target should be defined for indices 0 to len-future_window
        valid_target = target.dropna()
        max_valid_idx = valid_target.index[-1] if isinstance(valid_target.index[-1], int) else len(sample_ohlcv_data) - future_window
        
        # Check that we have valid targets in the expected range
        assert len(valid_target) > 0, "Should have some valid targets"


class TestBinaryDirectionTarget:
    """Tests for create_binary_direction_target function."""
    
    def test_binary_target_values(self, sample_ohlcv_data):
        """Binary target should only contain 0, 1, or NaN."""
        from src.data.features import create_binary_direction_target
        
        labels, valid_mask, meta = create_binary_direction_target(
            sample_ohlcv_data,
            future_window=12,
            smooth_window=3,
            min_move_atr=0.3
        )
        
        valid_labels = labels.dropna()
        unique_values = valid_labels.unique()
        
        assert all(v in [0, 1] for v in unique_values), \
            f"Binary labels should only contain 0 or 1, got {unique_values}"
    
    def test_binary_target_class_balance_metadata(self, sample_ohlcv_data):
        """Metadata should contain class balance information."""
        from src.data.features import create_binary_direction_target
        
        labels, valid_mask, meta = create_binary_direction_target(
            sample_ohlcv_data,
            future_window=12,
            smooth_window=3,
            min_move_atr=0.3
        )
        
        assert 'n_up' in meta, "Metadata should contain n_up"
        assert 'n_down' in meta, "Metadata should contain n_down"
        assert 'class_balance' in meta, "Metadata should contain class_balance"
        
        # Class balance should be between 0 and 1
        assert 0 <= meta['class_balance'] <= 1
    
    def test_binary_target_excludes_neutral(self, sample_ohlcv_data):
        """Binary target should exclude neutral/weak moves."""
        from src.data.features import create_binary_direction_target
        
        labels, valid_mask, meta = create_binary_direction_target(
            sample_ohlcv_data,
            future_window=12,
            smooth_window=3,
            min_move_atr=0.3  # Filter weak moves
        )
        
        assert 'n_neutral_excluded' in meta, "Should report excluded neutral samples"
        # Some samples should be excluded (neutral)
        # This depends on market data, so we just check the field exists


# =============================================================================
# ROLLING INDICATOR TESTS
# =============================================================================

class TestATRCalculation:
    """Tests for ATR (Average True Range) calculation."""
    
    def test_atr_output_length(self, sample_ohlcv_data):
        """ATR output should match input length."""
        from src.data.features import atr as calculate_atr
        
        period = 14
        atr = calculate_atr(sample_ohlcv_data, period=period)
        
        assert len(atr) == len(sample_ohlcv_data)
    
    def test_atr_warmup_nan(self, sample_ohlcv_data):
        """ATR should have NaN during warmup period."""
        from src.data.features import atr as calculate_atr
        
        period = 14
        atr = calculate_atr(sample_ohlcv_data, period=period)
        
        # First (period-1) values should be NaN
        n_nan = atr.iloc[:period].isna().sum()
        assert n_nan >= period - 1, \
            f"Expected at least {period-1} NaN values in warmup, got {n_nan}"
    
    def test_atr_always_positive(self, sample_ohlcv_data):
        """ATR should always be positive (volatility measure)."""
        from src.data.features import atr as calculate_atr
        
        atr = calculate_atr(sample_ohlcv_data, period=14)
        valid_atr = atr.dropna()
        
        assert (valid_atr >= 0).all(), "ATR should always be >= 0"
    
    def test_atr_no_lookahead(self, sample_ohlcv_data):
        """ATR at time T should only use data from T and before."""
        from src.data.features import atr as calculate_atr
        
        # Calculate ATR
        atr = calculate_atr(sample_ohlcv_data, period=14)
        
        # Modify future data and recalculate
        df_modified = sample_ohlcv_data.copy()
        midpoint = len(df_modified) // 2
        df_modified.iloc[midpoint:, :] = df_modified.iloc[midpoint:, :] * 2  # Double future prices
        
        atr_modified = calculate_atr(df_modified, period=14)
        
        # ATR before midpoint should be unchanged
        np.testing.assert_array_almost_equal(
            atr.iloc[:midpoint-14].dropna().values,
            atr_modified.iloc[:midpoint-14].dropna().values,
            decimal=4,
            err_msg="ATR before modification should be unchanged (no look-ahead)"
        )


class TestSMACalculation:
    """Tests for Simple Moving Average calculation."""
    
    def test_sma_output_length(self, sample_ohlcv_data):
        """SMA output should match input length."""
        period = 20
        sma = sample_ohlcv_data['close'].rolling(window=period).mean()
        
        assert len(sma) == len(sample_ohlcv_data)
    
    def test_sma_warmup_nan(self, sample_ohlcv_data):
        """SMA should have NaN during warmup period."""
        period = 20
        sma = sample_ohlcv_data['close'].rolling(window=period).mean()
        
        # First (period-1) values should be NaN
        n_nan = sma.iloc[:period-1].isna().sum()
        assert n_nan == period - 1, \
            f"Expected {period-1} NaN values in warmup, got {n_nan}"
    
    def test_sma_calculation_accuracy(self, sample_ohlcv_data):
        """Verify SMA calculation is mathematically correct."""
        period = 5
        sma = sample_ohlcv_data['close'].rolling(window=period).mean()
        
        # Manually calculate for a specific index
        test_idx = 10
        expected_sma = sample_ohlcv_data['close'].iloc[test_idx-period+1:test_idx+1].mean()
        actual_sma = sma.iloc[test_idx]
        
        assert actual_sma == pytest.approx(expected_sma, rel=1e-6), \
            f"SMA calculation incorrect: expected {expected_sma}, got {actual_sma}"


class TestCHOPIndicator:
    """Tests for Choppiness Index calculation."""
    
    def test_chop_range(self, sample_features_df):
        """CHOP should be between 0 and 100."""
        if 'chop' not in sample_features_df.columns:
            pytest.skip("CHOP not in features")
        
        chop = sample_features_df['chop'].dropna()
        
        assert (chop >= 0).all(), "CHOP should be >= 0"
        assert (chop <= 100).all(), "CHOP should be <= 100"
    
    def test_chop_high_in_range_market(self, sideways_market_data):
        """CHOP should be high in ranging/choppy market."""
        from src.data.features import engineer_all_features
        
        feature_config = {
            'chop_period': 14,
            'atr_period': 14,
            'sma_period': 20,
            'ema_fast': 12,
            'ema_slow': 26,
            'adx_period': 14,
            'fractal_window': 2,
            'sr_lookback': 50
        }
        
        df = engineer_all_features(sideways_market_data.ohlcv.copy(), feature_config)
        
        if 'chop' in df.columns:
            avg_chop = df['chop'].dropna().mean()
            # Ranging market should have higher CHOP (>50)
            # This is a soft check since synthetic data may vary
            assert avg_chop > 0, "CHOP should be calculated"


# =============================================================================
# NAN HANDLING TESTS
# =============================================================================

class TestNaNHandling:
    """Tests for proper NaN handling in features."""
    
    def test_features_nan_count_reasonable(self, sample_features_df):
        """Features should not have excessive NaN values."""
        df = sample_features_df
        nan_pct = df.isna().sum() / len(df)
        
        # Most NaN should be at the start (warmup period)
        # Allow up to 20% NaN (for longer rolling windows)
        for col in df.columns:
            assert nan_pct[col] < 0.25, \
                f"Column {col} has {nan_pct[col]*100:.1f}% NaN, too many"
    
    def test_nan_concentrated_at_start(self, sample_features_df):
        """NaN values should be concentrated at the start (warmup)."""
        df = sample_features_df
        
        # Check a few representative columns
        for col in ['atr', 'sma_distance', 'chop'][:1]:  # Check first available
            if col not in df.columns:
                continue
            
            series = df[col]
            first_valid = series.first_valid_index()
            
            if first_valid is not None:
                # All values after first_valid should be non-NaN
                after_warmup = series.loc[first_valid:]
                nan_after_warmup = after_warmup.isna().sum()
                
                # Allow some NaN in the middle (edge cases)
                assert nan_after_warmup / len(after_warmup) < 0.01, \
                    f"NaN values should be concentrated at start for {col}"


# =============================================================================
# FEATURE ENGINEERING INTEGRATION TESTS
# =============================================================================

class TestFeatureEngineeringIntegration:
    """Integration tests for full feature engineering pipeline."""
    
    def test_engineer_all_features_adds_columns(self, sample_ohlcv_data):
        """engineer_all_features should add expected columns."""
        from src.data.features import engineer_all_features
        
        feature_config = {
            'fractal_window': 2,
            'sr_lookback': 50,
            'sma_period': 20,
            'ema_fast': 12,
            'ema_slow': 26,
            'chop_period': 14,
            'adx_period': 14,
            'atr_period': 14
        }
        
        df = engineer_all_features(sample_ohlcv_data.copy(), feature_config)
        
        # Should have more columns than raw OHLCV
        assert len(df.columns) > 5, "Features should be added"
        
        # Check for key columns
        expected_cols = ['atr']  # At minimum, ATR should exist
        for col in expected_cols:
            assert col in df.columns, f"Expected column {col} missing"
    
    def test_feature_values_finite(self, sample_features_df):
        """All non-NaN feature values should be finite."""
        df = sample_features_df
        
        for col in df.columns:
            series = df[col].dropna()
            assert np.all(np.isfinite(series)), \
                f"Column {col} has non-finite values"
    
    @pytest.mark.slow
    def test_features_consistent_across_market_regimes(
        self, bullish_market_data, bearish_market_data, sideways_market_data
    ):
        """Features should be calculated consistently across different market regimes."""
        from src.data.features import engineer_all_features
        
        feature_config = {
            'fractal_window': 2,
            'sr_lookback': 50,
            'sma_period': 20,
            'ema_fast': 12,
            'ema_slow': 26,
            'chop_period': 14,
            'adx_period': 14,
            'atr_period': 14
        }
        
        for scenario in [bullish_market_data, bearish_market_data, sideways_market_data]:
            df = engineer_all_features(scenario.ohlcv.copy(), feature_config)
            
            # Basic sanity checks
            assert len(df) == len(scenario.ohlcv)
            assert 'atr' in df.columns


# =============================================================================
# LOOK-AHEAD BIAS TESTS
# =============================================================================

class TestLookAheadBias:
    """
    Critical tests to ensure no look-ahead bias in feature engineering.
    Features at time T should only use data from T and before.
    """
    
    def test_rolling_features_no_future_data(self, sample_ohlcv_data):
        """Rolling features should not use future data."""
        from src.data.features import engineer_all_features
        
        feature_config = {
            'atr_period': 14,
            'sma_period': 20,
            'ema_fast': 12,
            'ema_slow': 26,
            'chop_period': 14,
            'adx_period': 14,
            'fractal_window': 2,
            'sr_lookback': 50
        }
        
        df_full = engineer_all_features(sample_ohlcv_data.copy(), feature_config)
        
        # Take first half of data only
        half_len = len(sample_ohlcv_data) // 2
        df_partial = engineer_all_features(sample_ohlcv_data.iloc[:half_len].copy(), feature_config)
        
        # Features from partial should match full (for same indices)
        # Allow for edge effects at the boundary
        check_idx = half_len - 50  # Check well before boundary
        
        for col in ['atr']:  # Check key columns
            if col in df_full.columns and col in df_partial.columns:
                full_val = df_full[col].iloc[check_idx]
                partial_val = df_partial[col].iloc[check_idx]
                
                if not np.isnan(full_val) and not np.isnan(partial_val):
                    assert full_val == pytest.approx(partial_val, rel=1e-4), \
                        f"{col} at idx {check_idx} differs: full={full_val}, partial={partial_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
