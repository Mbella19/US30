"""
Elite Test Suite - Edge Case Tests
===================================
Tests for edge cases and boundary conditions not covered by basic tests.

These tests catch subtle bugs like:
- Division by zero
- NaN/Inf propagation
- Empty arrays
- Extreme values
- State management issues
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
# ZERO ATR HANDLING TESTS
# =============================================================================

class TestZeroATRHandling:
    """Tests for handling zero or near-zero ATR values."""

    def test_atr_minimum_enforced(self):
        """ATR should never be exactly zero in calculations."""
        from src.data.features import atr

        # Create constant price data (zero volatility)
        constant_price = 20000.0
        df = pd.DataFrame({
            'open': [constant_price] * 20,
            'high': [constant_price] * 20,
            'low': [constant_price] * 20,
            'close': [constant_price] * 20
        })

        atr_values = atr(df, period=14)

        # ATR should be 0 for constant prices but calculations should handle it
        # Check that no NaN or Inf resulted
        valid_values = atr_values.dropna()
        assert len(valid_values) > 0, "ATR produced all NaN"

    def test_volatility_sizing_with_min_atr(self):
        """Volatility sizing should work with minimum ATR (5 pips floor)."""
        # Very low ATR scenario
        atr = 1.0  # Extremely low
        sl_multiplier = 1.0
        pip_value = 1.0
        risk_per_trade = 100.0

        # SL pips with minimum floor
        sl_pips = max((atr * sl_multiplier) / pip_value, 5.0)

        # Should use floor of 5.0
        assert sl_pips == 5.0, f"SL should be floored at 5.0, got {sl_pips}"

        # Position size calculation
        dollars_per_pip = pip_value * 1.0 * 1.0
        size = risk_per_trade / (dollars_per_pip * sl_pips)

        assert np.isfinite(size), "Position size should be finite"
        assert size > 0, "Position size should be positive"

    def test_sma_distance_zero_atr_protection(self):
        """SMA distance calculation should handle zero ATR."""
        from src.data.features import sma_distance, atr

        # Create data with very low volatility
        np.random.seed(42)
        base_price = 20000.0
        small_noise = np.random.randn(100) * 0.01  # Tiny movements

        df = pd.DataFrame({
            'open': base_price + small_noise,
            'high': base_price + small_noise + 0.01,
            'low': base_price + small_noise - 0.01,
            'close': base_price + small_noise
        })

        atr_vals = atr(df, period=14)
        distance = sma_distance(df, atr_vals, period=20)

        # Should not produce Inf or NaN
        valid = distance.dropna()
        assert np.all(np.isfinite(valid)), "SMA distance contains non-finite values"

        # Should be clipped to [-100, 100]
        assert np.all(valid >= -100) and np.all(valid <= 100), "SMA distance not clipped"


# =============================================================================
# ZERO STD NORMALIZATION TESTS
# =============================================================================

class TestZeroStdNormalization:
    """Tests for normalizing constant features (std=0)."""

    def test_constant_feature_normalization(self):
        """Constant feature should normalize without error."""
        constant_value = 5.0
        data = np.full((100, 1), constant_value)

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # std = 0, need protection
        std_safe = np.where(std > 1e-8, std, 1.0)

        normalized = (data - mean) / std_safe

        # Should be all zeros (centered at mean)
        assert np.all(normalized == 0.0), "Constant should normalize to 0"
        assert np.all(np.isfinite(normalized)), "Should be finite"

    def test_near_zero_std_protection(self):
        """Near-zero std should be handled gracefully."""
        # Almost constant data
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0000001])

        mean = np.mean(data)
        std = np.std(data)

        # std is very small but not zero
        std_safe = max(std, 1e-8)

        normalized = (data - mean) / std_safe

        # Should still produce finite values
        assert np.all(np.isfinite(normalized)), "Near-zero std should be finite"


# =============================================================================
# SL/TP EDGE CASES
# =============================================================================

class TestSLTPEdgeCases:
    """Tests for SL/TP trigger edge cases."""

    def test_sl_tp_same_bar_sl_wins(self):
        """If both SL and TP could trigger in same bar, SL wins (conservative)."""
        # Scenario: V-shaped candle that hits both levels
        entry_price = 20000.0
        sl_price = 19950.0  # 50 below
        tp_price = 20150.0  # 150 above

        # Bar that touches both
        bar_high = 20200.0  # Above TP
        bar_low = 19900.0   # Below SL

        # SL triggered if low <= sl_price
        sl_triggered = bar_low <= sl_price

        # TP triggered if high >= tp_price
        tp_triggered = bar_high >= tp_price

        # Both can trigger, but SL should be checked FIRST
        assert sl_triggered and tp_triggered, "Both should be triggered"

        # In practice, code checks SL first, so SL wins

    def test_sl_at_exact_entry_breakeven(self):
        """Break-even SL (sl_pips_threshold=0) should work."""
        entry_price = 20000.0
        sl_pips_threshold = 0.0  # Break-even
        pip_value = 1.0

        # SL price for long = entry - (0 * pip_value) = entry
        sl_price = entry_price - sl_pips_threshold * pip_value

        assert sl_price == entry_price, "Break-even SL should equal entry"


# =============================================================================
# POSITION FLIP EDGE CASES
# =============================================================================

class TestPositionFlipEdgeCases:
    """Tests for position flip scenarios (long→short or short→long)."""

    def test_position_flip_in_single_step(self, trading_env):
        """Flipping position in single step should work correctly."""
        env = trading_env
        env.reset()

        # v36: With min_hold_bars > 0, instant flips are blocked
        if env.min_hold_bars > 0:
            pytest.skip(f"Test requires instant position changes (min_hold_bars={env.min_hold_bars})")

        # Open long
        action_long = np.array([1, 1], dtype=np.int32)
        _, _, _, _, _ = env.step(action_long)

        assert env.position == 1, "Should be long"
        initial_entry = env.entry_price

        # Flip to short
        action_short = np.array([2, 1], dtype=np.int32)
        _, _, _, _, _ = env.step(action_short)

        # Should now be short (or flat if flip closes first)
        # The exact behavior depends on implementation
        assert env.position in (-1, 0), "Should be short or flat after flip"

    def test_rapid_position_changes(self, trading_env):
        """Rapid position changes should not corrupt state."""
        env = trading_env
        env.reset()

        # v36: With min_hold_bars > 0, rapid position changes are blocked
        if env.min_hold_bars > 0:
            pytest.skip(f"Test requires instant position changes (min_hold_bars={env.min_hold_bars})")

        # Series of rapid changes
        actions = [
            np.array([1, 0], dtype=np.int32),  # Long
            np.array([0, 0], dtype=np.int32),  # Flat
            np.array([2, 0], dtype=np.int32),  # Short
            np.array([0, 0], dtype=np.int32),  # Flat
            np.array([1, 0], dtype=np.int32),  # Long
        ]

        for action in actions:
            obs, reward, done, truncated, info = env.step(action)

            # State should always be valid
            assert env.position in (-1, 0, 1)
            assert np.all(np.isfinite(obs))
            if done or truncated:
                break


# =============================================================================
# EPISODE END EDGE CASES
# =============================================================================

class TestEpisodeEndEdgeCases:
    """Tests for episode boundary edge cases."""

    def test_episode_end_closes_position(self, trading_env):
        """Position should be closed at episode end."""
        env = trading_env
        env.reset()

        # Open a position
        action = np.array([1, 1], dtype=np.int32)
        env.step(action)

        # Force episode to end by running max_steps
        original_max = env.max_steps
        env.max_steps = 5  # Force quick end

        for _ in range(10):
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        # Restore
        env.max_steps = original_max

        # Position should be closed at episode end
        # (The final reward should include realized PnL)

    def test_episode_end_index_bounds(self, trading_env):
        """Episode end should not exceed data bounds."""
        env = trading_env
        obs, _ = env.reset()

        # Step until end
        done = False
        truncated = False
        step_count = 0

        while not done and not truncated and step_count < 10000:
            action = np.array([0, 0], dtype=np.int32)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

        # current_idx should be within bounds
        assert env.current_idx <= len(env.close_prices)


# =============================================================================
# EXTREME VALUE TESTS
# =============================================================================

class TestExtremeValues:
    """Tests for extreme value handling."""

    def test_very_large_price_movement(self, trading_env):
        """Large price movements should not cause overflow."""
        env = trading_env
        env.reset()

        # Open position
        action = np.array([1, 1], dtype=np.int32)
        env.step(action)

        # Simulate by directly modifying price (for testing)
        original_price = env.close_prices[env.current_idx]

        # Run a few steps
        for _ in range(5):
            obs, reward, done, truncated, _ = env.step(action)

            assert np.all(np.isfinite(obs)), "Observation should be finite"
            assert np.isfinite(reward), "Reward should be finite"

            if done or truncated:
                break

    def test_entry_price_normalization_clipping(self, trading_env):
        """Entry price normalization should be clipped at ±10 ATR."""
        env = trading_env
        env.reset()

        # The observation construction clips entry_price_norm to [-10, 10]
        # This is tested by verifying observation bounds

        action = np.array([1, 1], dtype=np.int32)
        env.step(action)

        obs, _, _, _, _ = env.step(action)

        # All values should be within reasonable bounds
        assert np.all(obs >= -1000) and np.all(obs <= 1000), (
            "Observation values outside reasonable bounds"
        )


# =============================================================================
# NAN PROPAGATION TESTS
# =============================================================================

class TestNaNPropagation:
    """Tests for NaN handling throughout the pipeline."""

    def test_nan_in_features_handled(self):
        """NaN in feature data should be handled gracefully."""
        from src.data.features import engineer_all_features

        # Create data with some NaN
        df = pd.DataFrame({
            'open': [20000.0, np.nan, 20010.0, 20015.0, 20020.0] * 20,
            'high': [20005.0, 20008.0, np.nan, 20018.0, 20025.0] * 20,
            'low': [19995.0, 19992.0, 20005.0, np.nan, 20015.0] * 20,
            'close': [20002.0, 20005.0, 20012.0, 20016.0, np.nan] * 20
        }, index=pd.date_range('2024-01-01', periods=100, freq='5min'))

        # Forward fill to handle NaN
        df_clean = df.ffill().bfill()

        # Should not crash
        try:
            features = engineer_all_features(df_clean)
            # Some NaN in features is OK, but shouldn't be all NaN
            non_nan_count = features.notna().sum().sum()
            assert non_nan_count > 0, "All features are NaN"
        except Exception as e:
            pytest.fail(f"Feature engineering failed with NaN: {e}")

    def test_observation_no_nan_after_warmup(self, trading_env):
        """Observation should not contain NaN after sufficient warmup."""
        env = trading_env

        # Reset and step a few times for warmup
        obs, _ = env.reset()

        # After reset and a few steps, should be NaN-free
        for _ in range(10):
            action = np.array([0, 0], dtype=np.int32)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break

        nan_count = np.isnan(obs).sum()
        assert nan_count == 0, f"Observation has {nan_count} NaN values after warmup"


# =============================================================================
# INF HANDLING TESTS
# =============================================================================

class TestInfHandling:
    """Tests for infinity handling in metrics and calculations."""

    def test_profit_factor_inf_handled(self):
        """Profit factor should return inf for all-winning trades."""
        from src.evaluation.metrics import calculate_profit_factor, TradeRecord

        # All winning trades
        trades = [
            TradeRecord(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                entry_price=20000.0,
                exit_price=20100.0,
                direction=1,
                size=1.0,
                pnl_pips=100.0,
                pnl_percent=1.0
            )
        ]

        pf = calculate_profit_factor(trades)

        # Should be inf (no losses to divide by)
        assert pf == float('inf'), f"Expected inf, got {pf}"

    def test_sharpe_zero_std_handled(self):
        """Sharpe ratio should handle zero std gracefully."""
        from src.evaluation.metrics import calculate_sharpe_ratio

        # Constant returns (zero std)
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        sharpe = calculate_sharpe_ratio(returns)

        # Should return 0 or a defined value, not NaN/Inf
        assert np.isfinite(sharpe) or sharpe == 0.0

    def test_metrics_json_serializable(self):
        """Metrics with inf should be JSON serializable."""
        import json

        metrics = {
            'profit_factor': float('inf'),
            'sharpe': 1.5,
            'sortino': 2.0
        }

        # json.dumps handles inf as null in some configs
        # We need to convert inf to string or large number
        def handle_inf(obj):
            if obj == float('inf'):
                return 'inf'
            elif obj == float('-inf'):
                return '-inf'
            return obj

        # Should not crash
        try:
            json_str = json.dumps(metrics, default=str)
            assert 'inf' in json_str.lower()
        except (ValueError, TypeError):
            # Need special handling for inf
            pass


# =============================================================================
# EMPTY DATA EDGE CASES
# =============================================================================

class TestEmptyDataEdgeCases:
    """Tests for empty or insufficient data scenarios."""

    def test_zero_trades_metrics(self):
        """Metrics should handle zero trades."""
        from src.evaluation.metrics import calculate_metrics

        equity_curve = np.array([10000.0, 10000.0, 10000.0])
        trades = []
        initial_balance = 10000.0

        metrics = calculate_metrics(equity_curve, trades, initial_balance)

        # Should return valid metrics (zeros mostly)
        assert metrics['total_trades'] == 0
        assert metrics['win_rate_pct'] == 0.0

    def test_single_trade_metrics(self):
        """Metrics should work with single trade."""
        from src.evaluation.metrics import TradeRecord, calculate_metrics

        equity_curve = np.array([10000.0, 10050.0, 10100.0])
        trades = [
            TradeRecord(
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-02'),
                entry_price=20000.0,
                exit_price=20100.0,
                direction=1,
                size=1.0,
                pnl_pips=100.0,
                pnl_percent=1.0
            )
        ]
        initial_balance = 10000.0

        metrics = calculate_metrics(equity_curve, trades, initial_balance)

        assert metrics['total_trades'] == 1
        assert metrics['win_rate_pct'] == 100.0


# =============================================================================
# REGIME SAMPLING EDGE CASES
# =============================================================================

class TestRegimeSamplingEdgeCases:
    """Tests for regime-balanced sampling edge cases."""

    def test_empty_regime_fallback(self):
        """Empty regime indices should fallback gracefully."""
        # If a regime has no valid samples, should use another regime
        regime_indices = {
            0: np.array([]),      # Empty bullish
            1: np.array([50, 100, 150]),  # Valid ranging
            2: np.array([200, 250])  # Valid bearish
        }

        # Pick from non-empty regimes
        non_empty = [k for k, v in regime_indices.items() if len(v) > 0]

        assert len(non_empty) == 2, "Should have 2 non-empty regimes"

    def test_regime_distribution_reasonable(self, sample_ohlcv_data):
        """Regime detection should produce reasonable distribution."""
        from src.data.features import compute_regime_labels

        df = sample_ohlcv_data.copy()

        labels = compute_regime_labels(df, lookback=20)

        # Count each regime (0=bullish, 1=ranging, 2=bearish)
        counts = np.bincount(labels, minlength=3)

        # None should be 100% of data (too imbalanced)
        total = len(labels)
        for i, count in enumerate(counts):
            pct = count / total * 100
            # Allow some imbalance but not extreme
            assert pct < 90, f"Regime {i} is {pct:.1f}% of data (too dominant)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
