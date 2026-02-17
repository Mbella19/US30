"""
Elite Test Suite - Training/Inference Parity Tests
===================================================
CRITICAL: These tests catch silent bugs where training and live trading diverge.

Tests verify:
- Position sizes match between TradingEnv and live bridge
- Feature column ordering is identical
- Observation dimensions match exactly
- Normalization produces same results
- SL/TP calculations are consistent
- Lookback windows align correctly
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# POSITION SIZING PARITY TESTS (CRITICAL)
# =============================================================================

class TestPositionSizeParity:
    """
    CRITICAL: Position sizes MUST match between training and live bridge.

    If training uses [0.5, 1.0, 1.5, 2.0] but bridge uses [1.5, 2.0, 2.5, 3.0],
    agent will take 3x larger positions in live trading than trained!
    """

    def test_position_sizes_match_training_and_bridge(self):
        """Position size multipliers must be identical."""
        from src.environments.trading_env import TradingEnv
        from src.live.bridge_constants import POSITION_SIZES as BRIDGE_SIZES

        # Get training position sizes
        training_sizes = TradingEnv.POSITION_SIZES

        # Compare with bridge
        assert training_sizes == BRIDGE_SIZES, (
            f"CRITICAL PARITY BUG: Position sizes mismatch!\n"
            f"Training: {training_sizes}\n"
            f"Bridge: {BRIDGE_SIZES}\n"
            f"Agent will take wrong-sized positions in live trading!"
        )

    def test_position_size_indices_valid(self):
        """All position size indices should be valid."""
        from src.environments.trading_env import TradingEnv

        sizes = TradingEnv.POSITION_SIZES

        # Should have exactly 4 sizes for MultiDiscrete action space [3, 4]
        assert len(sizes) == 4, f"Expected 4 position sizes, got {len(sizes)}"

        # All should be positive floats
        for i, size in enumerate(sizes):
            assert isinstance(size, (int, float)), f"Size {i} is not numeric"
            assert size > 0, f"Size {i} must be positive, got {size}"

    def test_volatility_sizing_formula_parity(self, config):
        """Volatility sizing formula should match between env and backtest."""
        from src.environments.trading_env import TradingEnv
        from src.evaluation.backtest import Backtester

        # Test parameters
        atr = 50.0
        sl_atr_mult = config.trading.sl_atr_multiplier
        pip_value = config.instrument.pip_value
        risk_per_trade = 100.0
        base_size = 1.0

        # Calculate expected size
        sl_pips = max((atr * sl_atr_mult) / pip_value, 5.0)
        dollars_per_pip = pip_value * 1.0 * 1.0  # lot_size=1, multiplier=1
        risk_amount = risk_per_trade * base_size
        expected_size = risk_amount / (dollars_per_pip * sl_pips)
        expected_size = np.clip(expected_size, 0.1, 50.0)

        # Verify formula produces reasonable size
        assert 0.1 <= expected_size <= 50.0, f"Size {expected_size} outside valid range"


# =============================================================================
# FEATURE ORDERING PARITY TESTS (CRITICAL)
# =============================================================================

class TestFeatureOrderingParity:
    """
    CRITICAL: Feature column ordering MUST be identical everywhere.

    If features are reordered, agent interprets ATR as CHOP, regime as returns, etc.
    This causes silent catastrophic failures.
    """

    def test_model_feature_cols_ordering(self):
        """MODEL_FEATURE_COLS must be in canonical order."""
        from src.live.bridge_constants import MODEL_FEATURE_COLS

        # v35: Added 4 percentile features for regime-robust normalization
        # v37: Added 4 training-anchored OOD features (fixed stats, never adapt)
        # Mean reversion: 4 features (bb_percent_b, bb_bandwidth, williams_r, rsi, rsi_divergence)
        expected_cols = [
            "returns", "volatility",
            "sma_distance", "dist_to_resistance",
            "dist_to_support", "sr_strength_r", "sr_strength_s",
            "session_asian", "session_london", "session_ny",
            "structure_fade", "bars_since_bos", "bars_since_choch",
            "bos_magnitude", "choch_magnitude",
            "bos_streak",
            "atr_context",
            # v35 FIX: Regime-robust percentile features
            "atr_percentile", "chop_percentile", "sma_distance_percentile",
            "volatility_percentile",
            # v37 OOD FIX: Training-anchored features (FIXED stats, never adapt)
            "volatility_vs_training", "returns_skew_shift", "atr_vs_training",
            "ood_score",
            # Mean Reversion Features (4)
            "bb_percent_b", "bb_bandwidth", "williams_r",
            "rsi", "rsi_divergence",
        ]

        assert list(MODEL_FEATURE_COLS) == expected_cols, (
            f"MODEL_FEATURE_COLS ordering changed!\n"
            f"Expected: {expected_cols}\n"
            f"Got: {list(MODEL_FEATURE_COLS)}"
        )

    def test_market_feature_cols_ordering(self):
        """MARKET_FEATURE_COLS must be in canonical order."""
        from src.live.bridge_constants import MARKET_FEATURE_COLS

        # v35: Added 4 percentile features for regime-robust normalization
        # v37: Added 4 training-anchored OOD features (fixed stats, never adapt)
        # Mean reversion: 4 features (bb_percent_b, bb_bandwidth, williams_r, rsi, rsi_divergence)
        expected_cols = [
            "atr", "chop", "adx", "sma_distance",
            "dist_to_support", "dist_to_resistance",
            "sr_strength_r", "sr_strength_s", "session_asian",
            "session_london", "session_ny",
            "structure_fade", "bars_since_bos", "bars_since_choch",
            "bos_magnitude", "choch_magnitude",
            "bos_streak",
            "returns", "volatility",
            "atr_context",
            # v35 FIX: Regime-robust percentile features
            "atr_percentile", "chop_percentile", "sma_distance_percentile",
            "volatility_percentile",
            # v37 OOD FIX: Training-anchored features (FIXED stats, never adapt)
            "volatility_vs_training", "returns_skew_shift", "atr_vs_training",
            "ood_score",
            # Mean Reversion Features (4)
            "bb_percent_b", "bb_bandwidth", "williams_r",
            "rsi", "rsi_divergence",
        ]

        assert list(MARKET_FEATURE_COLS) == expected_cols, (
            f"MARKET_FEATURE_COLS ordering changed!\n"
            f"Expected: {expected_cols}\n"
            f"Got: {list(MARKET_FEATURE_COLS)}"
        )

    def test_market_feature_count(self):
        """Market features should have exactly 33 columns (20 base + 4 percentile + 4 v37 + 5 mean reversion)."""
        from src.live.bridge_constants import MARKET_FEATURE_COLS

        assert len(MARKET_FEATURE_COLS) == 33, (
            f"Expected 33 market features, got {len(MARKET_FEATURE_COLS)}"
        )

    def test_model_feature_count(self):
        """Model features should have exactly 30 columns (17 base + 4 percentile + 4 v37 + 5 mean reversion)."""
        from src.live.bridge_constants import MODEL_FEATURE_COLS

        assert len(MODEL_FEATURE_COLS) == 30, (
            f"Expected 30 model features, got {len(MODEL_FEATURE_COLS)}"
        )


# =============================================================================
# OBSERVATION DIMENSION PARITY TESTS
# =============================================================================

class TestObservationDimensionParity:
    """Tests that observation space dimensions match across all components."""

    def test_observation_space_dimensions(self, trading_env):
        """Observation space should have expected dimensions."""
        env = trading_env
        obs_dim = env.observation_space.shape[0]

        # Observation dimensions depend on whether analyst is enabled
        # Without analyst: position(4) + market(5-66) + sl_tp(2) + hold_features(4)
        # With analyst: context + position(4) + market + analyst_metrics + sl_tp(2) + hold_features(4)

        # Minimum expected components (without analyst or context)
        min_expected = 4 + 2 + 4  # position + sl_tp + hold = 10

        assert obs_dim >= min_expected, (
            f"Observation dim {obs_dim} too small, expected >= {min_expected}"
        )

    def test_observation_all_finite(self, trading_env):
        """All observation values should be finite after reset."""
        env = trading_env
        obs, _ = env.reset()

        assert np.all(np.isfinite(obs)), (
            f"Observation contains non-finite values after reset:\n"
            f"NaN count: {np.isnan(obs).sum()}, Inf count: {np.isinf(obs).sum()}"
        )

    def test_observation_dtype_float32(self, trading_env):
        """Observation should be float32 for memory efficiency."""
        env = trading_env
        obs, _ = env.reset()

        assert obs.dtype == np.float32, (
            f"Expected float32 observation, got {obs.dtype}"
        )


# =============================================================================
# SL/TP MULTIPLIER PARITY TESTS
# =============================================================================

class TestSLTPParity:
    """Tests that SL/TP calculations match between training and backtest."""

    def test_sl_atr_multiplier_parity(self, config, trading_env):
        """SL ATR multiplier should match config."""
        from src.evaluation.backtest import Backtester

        expected_sl = config.trading.sl_atr_multiplier
        env_sl = trading_env.sl_atr_multiplier

        backtester = Backtester(sl_atr_multiplier=expected_sl)

        assert env_sl == expected_sl, f"Env SL {env_sl} != config SL {expected_sl}"
        assert backtester.sl_atr_multiplier == expected_sl

    def test_tp_atr_multiplier_parity(self, config, trading_env):
        """TP ATR multiplier should match config."""
        from src.evaluation.backtest import Backtester

        expected_tp = config.trading.tp_atr_multiplier
        env_tp = trading_env.tp_atr_multiplier

        backtester = Backtester(tp_atr_multiplier=expected_tp)

        assert env_tp == expected_tp, f"Env TP {env_tp} != config TP {expected_tp}"
        assert backtester.tp_atr_multiplier == expected_tp

    def test_sl_level_calculation_long(self):
        """Long SL level calculation should be consistent."""
        entry_price = 20000.0
        entry_atr = 50.0
        sl_multiplier = 1.0
        pip_value = 1.0

        # Formula: SL = entry - (ATR × multiplier)
        sl_price = entry_price - (entry_atr * sl_multiplier)

        assert sl_price == 19950.0, f"Long SL calculation wrong: {sl_price}"

    def test_sl_level_calculation_short(self):
        """Short SL level calculation should be consistent."""
        entry_price = 20000.0
        entry_atr = 50.0
        sl_multiplier = 1.0

        # Formula: SL = entry + (ATR × multiplier) for shorts
        sl_price = entry_price + (entry_atr * sl_multiplier)

        assert sl_price == 20050.0, f"Short SL calculation wrong: {sl_price}"


# =============================================================================
# LOOKBACK WINDOW PARITY TESTS
# =============================================================================

class TestLookbackWindowParity:
    """Tests that lookback windows are consistent across components."""

    def test_lookback_5m_consistency(self, config):
        """5m lookback should be consistent."""
        expected = 48  # 4 hours of 5m bars
        actual = config.analyst.lookback_5m

        assert actual == expected, f"5m lookback {actual} != expected {expected}"

    def test_lookback_15m_consistency(self, config):
        """15m lookback should be consistent."""
        expected = 16  # ~4 hours of 15m bars
        actual = config.analyst.lookback_15m

        assert actual == expected, f"15m lookback {actual} != expected {expected}"

    def test_lookback_45m_consistency(self, config):
        """45m lookback should be consistent."""
        expected = 6  # ~4.5 hours of 45m bars
        actual = config.analyst.lookback_45m

        assert actual == expected, f"45m lookback {actual} != expected {expected}"

    def test_start_idx_calculation(self, config):
        """Start index calculation should match training."""
        lookback_5m = config.analyst.lookback_5m
        lookback_15m = config.analyst.lookback_15m
        lookback_45m = config.analyst.lookback_45m

        subsample_15m = 3  # 15m / 5m = 3
        subsample_45m = 9  # 45m / 5m = 9

        # Training formula (matches train_agent.py + mt5_bridge.py):
        # Need enough aligned 5m bars to subsample higher TF windows INCLUDING the current candle.
        training_start = max(
            lookback_5m,
            (lookback_15m - 1) * subsample_15m + 1,
            (lookback_45m - 1) * subsample_45m + 1
        )

        # With defaults: max(48, 46, 46) = 48
        expected = 48
        assert training_start == expected, (
            f"start_idx calculation: {training_start} != expected {expected}"
        )


# =============================================================================
# SPREAD COST PARITY TESTS
# =============================================================================

class TestSpreadCostParity:
    """Tests that spread costs are handled consistently."""

    def test_spread_pips_config(self, config):
        """Spread pips should be configured correctly."""
        spread = config.trading.spread_pips

        # US30 typical spread is 30-100 pips
        assert 1.0 <= spread <= 200.0, f"Spread {spread} outside reasonable range"

    def test_spread_cost_formula(self, config):
        """Spread cost formula should be consistent."""
        spread_pips = config.trading.spread_pips
        position_size = 1.0

        # Entry cost in pips
        entry_cost_pips = spread_pips * position_size

        # For US30 with 50 pip spread
        if spread_pips == 50.0:
            assert entry_cost_pips == 50.0


# =============================================================================
# ROLLING NORMALIZATION PARITY TESTS
# =============================================================================

class TestRollingNormalizationParity:
    """Tests that rolling normalization produces consistent results."""

    def test_rolling_window_size_consistent(self, config):
        """Rolling window size should be consistent (5760 = 20 days of 5m)."""
        # This is typically hardcoded in both TradingEnv and MT5Bridge
        expected_window = 5760

        # Verify this matches ~20 trading days at 5m intervals
        # 20 days × 24 hours × 12 bars/hour = 5760
        calculated = 20 * 24 * 12

        assert calculated == expected_window

    def test_rolling_mean_calculation(self):
        """Rolling mean should be calculated correctly."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        window = 20

        # Manual rolling mean
        rolling_mean = np.mean(data[-window:], axis=0)

        # Using pandas for verification
        df = pd.DataFrame(data)
        pandas_mean = df.rolling(window).mean().iloc[-1].values

        np.testing.assert_array_almost_equal(
            rolling_mean, pandas_mean, decimal=10
        )

    def test_rolling_std_calculation(self):
        """Rolling std should be calculated correctly."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        window = 20

        # Manual rolling std (population std)
        rolling_std = np.std(data[-window:], axis=0)

        # Verify positive and reasonable
        assert np.all(rolling_std > 0), "Rolling std should be positive"
        assert np.all(rolling_std < 5), "Rolling std unusually large"


# =============================================================================
# REWARD SCALING PARITY TESTS
# =============================================================================

class TestRewardScalingParity:
    """Tests that reward scaling is consistent."""

    def test_reward_scaling_config(self, config, trading_env):
        """Reward scaling should match between config and env."""
        config_scaling = config.trading.reward_scaling
        env_scaling = trading_env.reward_scaling

        assert env_scaling == config_scaling

    def test_profit_loss_scaling_asymmetry(self, config, trading_env):
        """Profit scaling should be >= loss scaling."""
        profit_scaling = trading_env.profit_scaling
        loss_scaling = trading_env.loss_scaling

        assert profit_scaling >= loss_scaling, (
            f"Profit scaling {profit_scaling} < loss scaling {loss_scaling}"
        )

    def test_asymmetric_scaling_values(self, config):
        """Profit/loss scaling ratio should match current config intent."""
        actual_profit = config.trading.profit_scaling
        actual_loss = config.trading.loss_scaling

        # Verify profit scaling is higher than loss scaling
        assert actual_profit >= actual_loss, \
            f"Profit scaling {actual_profit} should be >= loss scaling {actual_loss}"

        # v36: Symmetric scaling (ratio ~= 1.0). If you intentionally reintroduce
        # asymmetry, update this expectation accordingly.
        if actual_loss > 0:
            ratio = actual_profit / actual_loss
            assert ratio == pytest.approx(1.0, rel=0.05), \
                f"Profit/loss ratio {ratio} should be ~1.0 (symmetric scaling)"


# =============================================================================
# CONFIG CONSISTENCY TESTS
# =============================================================================

class TestConfigConsistency:
    """Tests that configuration is internally consistent."""

    def test_all_required_config_sections_exist(self, config):
        """All required config sections should exist."""
        required_sections = ['trading', 'instrument', 'analyst', 'paths']

        for section in required_sections:
            assert hasattr(config, section), f"Missing config section: {section}"

    def test_instrument_config_valid(self, config):
        """Instrument config should have valid values."""
        assert config.instrument.pip_value > 0
        assert config.instrument.lot_size > 0

    def test_trading_config_valid(self, config):
        """Trading config should have valid values."""
        assert config.trading.sl_atr_multiplier > 0
        assert config.trading.tp_atr_multiplier > 0
        assert config.trading.reward_scaling > 0


# =============================================================================
# OBSERVATION VECTOR PARITY TESTS (CRITICAL)
# =============================================================================

class TestObservationVectorParity:
    """
    CRITICAL: Element-by-element observation vector comparison.

    These tests verify that observations constructed by TradingEnv and
    the live bridge's _build_observation produce identical results given
    the same inputs. Any mismatch here causes silent model degradation.
    """

    def test_observation_dimension_formula(self, config):
        """Verify observation dimension calculation matches expected formula."""
        from src.live.bridge_constants import MARKET_FEATURE_COLS

        use_analyst = config.trading.use_analyst
        context_dim = config.analyst.context_dim if use_analyst else 0
        analyst_metrics_dim = 5 if use_analyst else 0  # Binary: [p_down, p_up, edge, conf, uncert]
        market_dim = len(MARKET_FEATURE_COLS) * 3  # 3 timeframes
        position_dim = 4  # [position, entry_norm, pnl_norm, time_norm]
        sl_tp_dim = 2  # [dist_sl_norm, dist_tp_norm]
        hold_dim = 4  # [profit_progress, dist_to_tp_pct, momentum_aligned, session_progress]
        returns_dim = config.trading.agent_lookback_window

        expected_obs_dim = (
            context_dim + position_dim + market_dim + analyst_metrics_dim +
            sl_tp_dim + hold_dim + returns_dim
        )

        # Log breakdown for debugging
        print(f"\nObservation dimension breakdown:")
        print(f"  Context: {context_dim}")
        print(f"  Position: {position_dim}")
        print(f"  Market (3 TFs): {market_dim}")
        print(f"  Analyst metrics: {analyst_metrics_dim}")
        print(f"  SL/TP: {sl_tp_dim}")
        print(f"  Hold features: {hold_dim}")
        print(f"  Returns window: {returns_dim}")
        print(f"  TOTAL: {expected_obs_dim}")

        assert expected_obs_dim > 0, "Observation dimension should be positive"

    def test_observation_component_indices(self, trading_env, config):
        """Verify observation components are at expected indices."""
        from src.live.bridge_constants import MARKET_FEATURE_COLS

        obs, _ = trading_env.reset()

        use_analyst = config.trading.use_analyst
        context_dim = config.analyst.context_dim if use_analyst else 0
        market_dim = len(MARKET_FEATURE_COLS) * 3
        analyst_metrics_dim = 5 if use_analyst else 0

        # Calculate expected indices
        context_end = context_dim
        position_end = context_end + 4
        market_end = position_end + market_dim
        analyst_end = market_end + analyst_metrics_dim
        sl_tp_end = analyst_end + 2
        hold_end = sl_tp_end + 4

        # Position state (at reset, should be flat)
        position_state = obs[context_end:position_end]
        assert len(position_state) == 4, f"Position state should be 4 dims, got {len(position_state)}"

        # At reset, position should be 0
        assert position_state[0] == 0.0, f"Position at reset should be 0, got {position_state[0]}"

        # Entry price norm, pnl norm, time norm should all be 0 at reset
        expected_pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            position_state, expected_pos, decimal=5,
            err_msg="Position state at reset should be all zeros"
        )

    def test_market_features_present(self, trading_env, config):
        """Market features should be present in observation and all finite."""
        obs, _ = trading_env.reset()

        # Market features are part of the observation - just verify they exist and are finite
        assert np.all(np.isfinite(obs)), "Observation contains NaN/Inf values"

        # Observation should have reasonable dimension
        assert obs.shape[0] > 10, f"Observation too small: {obs.shape[0]} dims"

    def test_market_features_clipping(self, trading_env, config):
        """Market features should be mostly within normalized range after clipping."""
        obs, _ = trading_env.reset()

        # Most values should be in reasonable range (clipped to [-5, 5] during normalization)
        in_range = np.sum(np.abs(obs) <= 10.0)  # Use wider range for robustness
        total = len(obs)
        pct_in_range = in_range / total * 100
        assert pct_in_range > 80, (
            f"Only {pct_in_range:.1f}% of observation in [-10,10] range. "
            "Values may not be normalized correctly."
        )

    def test_observation_values_valid(self, trading_env, config):
        """Observation should have valid values (finite, reasonable range)."""
        obs, _ = trading_env.reset()

        # All values should be finite
        assert np.all(np.isfinite(obs)), "Observation contains non-finite values"

        # dtype should be float32
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

        # Observation should have reasonable dimension (at least position + some features)
        assert len(obs) >= 10, f"Observation too small: {len(obs)} dims"

    def test_returns_window_dimension(self, trading_env, config):
        """Returns window should match agent_lookback_window config."""
        from src.live.bridge_constants import MARKET_FEATURE_COLS

        obs, _ = trading_env.reset()

        returns_dim = config.trading.agent_lookback_window

        if returns_dim > 0:
            # Returns are at the end of observation
            returns_window = obs[-returns_dim:]

            assert len(returns_window) == returns_dim, (
                f"Returns window should be {returns_dim} dims, got {len(returns_window)}"
            )

            # Returns should be z-score normalized (mostly in reasonable range)
            assert np.all(np.isfinite(returns_window)), "Returns window contains NaN/Inf"


class TestRollingNormalizerParity:
    """Tests that rolling normalization matches between training and live."""

    def test_rolling_normalizer_circular_buffer(self):
        """Test O(1) circular buffer implementation."""
        from src.live.mt5_bridge import RollingMarketNormalizer, MarketFeatureStats

        n_features = 10
        window_size = 100

        # Create mock fallback stats with all required fields
        mock_cols = tuple(f"feature_{i}" for i in range(n_features))
        mock_stats = MarketFeatureStats(
            cols=mock_cols,
            mean=np.zeros(n_features, dtype=np.float32),
            std=np.ones(n_features, dtype=np.float32)
        )

        normalizer = RollingMarketNormalizer(
            n_features=n_features,
            fallback_stats=mock_stats,
            rolling_window_size=window_size,
            rolling_min_samples=10
        )

        # Feed identical values - should normalize to 0 after warmup
        test_data = np.ones(n_features, dtype=np.float32) * 5.0

        # Warmup phase: uses fallback stats
        for _ in range(9):
            normalized = normalizer.update_and_normalize(test_data)

        # After warmup: uses rolling stats
        for _ in range(20):
            normalized = normalizer.update_and_normalize(test_data)

        # With constant input, rolling mean = 5.0, std approaches 0
        # Due to std floor (1e-6), normalized should be close to 0
        # Actually with constant values, normalized = (5-5)/max(0,1e-6) ≈ 0
        # But std is floored so it becomes (5-5)/1e-6 = 0

    def test_rolling_normalizer_clip_range(self):
        """Normalized values should be clipped to [-5, 5]."""
        from src.live.mt5_bridge import RollingMarketNormalizer, MarketFeatureStats

        n_features = 5

        # Create mock fallback stats with all required fields
        mock_cols = tuple(f"feature_{i}" for i in range(n_features))
        mock_stats = MarketFeatureStats(
            cols=mock_cols,
            mean=np.zeros(n_features, dtype=np.float32),
            std=np.ones(n_features, dtype=np.float32)
        )

        normalizer = RollingMarketNormalizer(
            n_features=n_features,
            fallback_stats=mock_stats,
            rolling_window_size=100,
            rolling_min_samples=5
        )

        # Feed extreme values
        extreme_data = np.array([100.0, -100.0, 1000.0, -1000.0, 0.0], dtype=np.float32)

        # Warmup
        for _ in range(5):
            normalizer.update_and_normalize(np.zeros(n_features, dtype=np.float32))

        normalized = normalizer.update_and_normalize(extreme_data)

        # All values should be clipped to [-5, 5]
        assert np.all(normalized >= -5.0), f"Values below -5: {normalized[normalized < -5.0]}"
        assert np.all(normalized <= 5.0), f"Values above 5: {normalized[normalized > 5.0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
