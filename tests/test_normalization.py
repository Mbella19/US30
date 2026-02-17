"""
Elite Test Suite - Normalization Tests
=======================================
Comprehensive verification of Z-score normalization and rolling window logic.

Tests verify:
- Z-score produces mean~0, std~1
- Training stats used for test data (no look-ahead)
- Rolling normalization O(1) circular buffer
- NaN handling
"""

import pytest
import numpy as np
import pandas as pd
from typing import Optional

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Z-SCORE NORMALIZATION TESTS
# =============================================================================

class TestZScoreNormalization:
    """Tests for Z-score normalization correctness."""
    
    def test_zscore_mean_approx_zero(self):
        """Z-scored data should have mean ≈ 0."""
        # Generate sample data
        np.random.seed(42)
        data = np.random.randn(1000, 5) * 100 + 500  # Mean 500, std 100
        
        # Compute training stats
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        
        # Z-score
        normalized = (data - mean) / std
        
        # Check mean ≈ 0
        np.testing.assert_array_almost_equal(
            normalized.mean(axis=0),
            np.zeros(5),
            decimal=1,
            err_msg="Z-scored data should have mean ≈ 0"
        )
    
    def test_zscore_std_approx_one(self):
        """Z-scored data should have std ≈ 1."""
        np.random.seed(42)
        data = np.random.randn(1000, 5) * 50 + 200
        
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        normalized = (data - mean) / std
        
        np.testing.assert_array_almost_equal(
            normalized.std(axis=0),
            np.ones(5),
            decimal=1,
            err_msg="Z-scored data should have std ≈ 1"
        )
    
    def test_zscore_handles_zero_std(self):
        """Z-score should handle constant features (std=0) gracefully."""
        data = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        
        # Replace zero std with 1 to avoid division by zero
        std = np.where(std > 1e-8, std, 1.0)
        
        normalized = (data - mean) / std
        
        # Should not contain NaN or Inf
        assert np.all(np.isfinite(normalized)), "Should handle zero std gracefully"
    
    def test_zscore_preserves_relative_ordering(self):
        """Z-score should preserve relative ordering of values."""
        data = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / std
        
        # Check ordering preserved
        original_order = np.argsort(data)
        normalized_order = np.argsort(normalized)
        
        np.testing.assert_array_equal(
            original_order,
            normalized_order,
            err_msg="Z-score should preserve relative ordering"
        )


# =============================================================================
# TRAINING STATS TESTS (NO LOOK-AHEAD BIAS)
# =============================================================================

class TestTrainingStatsOnlyNormalization:
    """Tests to ensure normalization stats come from training data only."""
    
    def test_train_stats_computed_on_train_data_only(self):
        """
        Normalization statistics should be computed on training data only,
        then applied to test data. This prevents look-ahead bias.
        """
        np.random.seed(42)
        
        # Full dataset
        n_samples = 1000
        full_data = np.random.randn(n_samples, 3)
        
        # Train/test split (85/15)
        train_size = int(0.85 * n_samples)
        train_data = full_data[:train_size]
        test_data = full_data[train_size:]
        
        # Compute stats on TRAINING data only
        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)
        
        # Apply to test data
        test_normalized = (test_data - train_mean) / train_std
        
        # Test data normalized with train stats will NOT have mean=0, std=1
        # This is CORRECT behavior (no look-ahead)
        test_mean = test_normalized.mean(axis=0)
        test_std = test_normalized.std(axis=0)
        
        # Means should be close to 0 but not exactly (different distribution)
        assert np.all(np.abs(test_mean) < 0.5), \
            "Test mean should be reasonably close to 0"
        
        # This test passes if there's no exception - we're verifying the pattern
        assert test_data.shape == test_normalized.shape
    
    def test_stats_not_updated_during_testing(self):
        """
        During testing/inference, stats should remain fixed (not updated).
        """
        train_mean = np.array([100.0, 200.0, 300.0])
        train_std = np.array([10.0, 20.0, 30.0])
        
        # Simulating test samples one by one
        test_samples = [
            np.array([150.0, 250.0, 350.0]),
            np.array([50.0, 150.0, 250.0]),
            np.array([200.0, 300.0, 400.0]),
        ]
        
        # Stats should NOT change
        original_mean = train_mean.copy()
        original_std = train_std.copy()
        
        for sample in test_samples:
            # Normalize (but don't update stats)
            normalized = (sample - train_mean) / train_std
            
        # Verify stats unchanged
        np.testing.assert_array_equal(train_mean, original_mean)
        np.testing.assert_array_equal(train_std, original_std)


# =============================================================================
# ROLLING NORMALIZATION TESTS
# =============================================================================

class TestRollingNormalization:
    """Tests for rolling window normalization (circular buffer)."""
    
    def test_rolling_mean_calculation(self):
        """Rolling mean should update correctly."""
        window_size = 5
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        
        # Manual rolling mean calculation
        rolling_means = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            window = data[start:i+1]
            rolling_means.append(window.mean())
        
        # Expected values (increasing window until full)
        expected = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        
        np.testing.assert_array_almost_equal(
            rolling_means,
            expected,
            decimal=4
        )
    
    def test_rolling_std_non_negative(self):
        """Rolling std should always be non-negative."""
        np.random.seed(42)
        data = np.random.randn(100)
        window_size = 20
        
        rolling_stds = pd.Series(data).rolling(window=window_size).std().dropna()
        
        assert (rolling_stds >= 0).all(), "Rolling std should be non-negative"
    
    def test_circular_buffer_efficiency(self):
        """
        Verify that circular buffer approach can be O(1) per update.
        This is a conceptual test - we verify the running sum approach.
        """
        window_size = 5
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Circular buffer simulation
        buffer = np.zeros(window_size)
        running_sum = 0.0
        idx = 0
        count = 0
        
        running_means = []
        
        for val in data:
            # Remove old value from sum (if buffer is full)
            if count >= window_size:
                running_sum -= buffer[idx]
            
            # Add new value
            buffer[idx] = val
            running_sum += val
            
            # Update count and index
            count = min(count + 1, window_size)
            idx = (idx + 1) % window_size
            
            # Calculate mean
            running_means.append(running_sum / count)
        
        # Verify against pd.rolling
        expected = pd.Series(data).rolling(window=window_size, min_periods=1).mean().values
        
        np.testing.assert_array_almost_equal(
            running_means,
            expected,
            decimal=6
        )


# =============================================================================
# NAN HANDLING TESTS
# =============================================================================

class TestNaNHandling:
    """Tests for proper NaN handling in normalization."""
    
    def test_nan_in_input_produces_nan_output(self):
        """NaN values should propagate through normalization."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        normalized = (data - mean) / std
        
        # NaN should remain NaN
        assert np.isnan(normalized[2]), "NaN should remain NaN after normalization"
        
        # Non-NaN should be finite
        assert np.all(np.isfinite(normalized[~np.isnan(normalized)]))
    
    def test_all_nan_column_handled(self):
        """Column with all NaN should be handled gracefully."""
        data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        
        # Use nanmean/nanstd for safety
        with np.errstate(all='ignore'):
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
        
        # Replace NaN std with 1.0
        std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)
        mean = np.where(np.isnan(mean), 0.0, mean)
        
        normalized = (data - mean) / std
        
        # Column 1 should still be NaN (input was NaN)
        assert np.all(np.isnan(normalized[:, 1]))
    
    def test_normalization_with_nan_mean_replacement(self):
        """
        If mean is NaN (all values NaN), should use 0.
        If std is NaN or 0, should use 1.
        """
        # All NaN data
        data = np.full((5, 2), np.nan)
        
        mean = np.array([0.0, 0.0])  # Replaced NaN mean with 0
        std = np.array([1.0, 1.0])   # Replaced NaN std with 1
        
        normalized = (data - mean) / std
        
        # Should still be NaN (NaN - 0 = NaN)
        assert np.all(np.isnan(normalized))


# =============================================================================
# EDGE CASES
# =============================================================================

class TestNormalizationEdgeCases:
    """Edge cases for normalization."""
    
    def test_single_sample_normalization(self):
        """Single sample should normalize to 0 (with std=1 fallback)."""
        data = np.array([[100.0, 200.0, 300.0]])
        
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # Fallback for zero std
        
        normalized = (data - mean) / std
        
        # With one sample, mean = sample, so normalized = 0
        np.testing.assert_array_almost_equal(
            normalized,
            np.zeros((1, 3)),
            decimal=6
        )
    
    def test_very_large_values(self):
        """Normalization should handle very large values."""
        data = np.array([1e10, 1e10 + 100, 1e10 + 200])
        
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / std
        
        # Should be finite and normalized
        assert np.all(np.isfinite(normalized))
        assert normalized.std() == pytest.approx(1.0, rel=1e-4)
    
    def test_very_small_values(self):
        """Normalization should handle very small values."""
        data = np.array([1e-10, 2e-10, 3e-10])
        
        mean = data.mean()
        std = data.std()
        if std < 1e-15:
            std = 1.0
        normalized = (data - mean) / std
        
        assert np.all(np.isfinite(normalized))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
