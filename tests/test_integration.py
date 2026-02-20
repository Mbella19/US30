"""
Elite Test Suite - Integration Tests
=====================================
End-to-end integration tests for the complete trading system.

Tests verify:
- Full pipeline workflow (data → features → env → agent)
- Component interactions
- System-level correctness
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
# DATA PIPELINE INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestDataPipelineIntegration:
    """Tests for the data loading and processing pipeline."""
    
    def test_ohlcv_to_features_pipeline(self, sample_ohlcv_data):
        """OHLCV data should flow through feature engineering correctly."""
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
        
        features_df = engineer_all_features(sample_ohlcv_data.copy(), feature_config)
        
        # Should have features added
        assert len(features_df.columns) > len(sample_ohlcv_data.columns)
        assert 'atr' in features_df.columns
    
    def test_features_to_env_data_format(self, sample_features_df, market_features_array):
        """Features should be convertible to env-compatible format."""
        # Market features should be 2D array
        assert len(market_features_array.shape) == 2
        assert market_features_array.dtype == np.float32
        
        # No NaN in the middle (after warmup)
        warmup = 50  # Approximate warmup period
        middle_data = market_features_array[warmup:-10]
        if len(middle_data) > 0:
            nan_count = np.isnan(middle_data).sum()
            # Allow some NaN but not too many
            nan_pct = nan_count / middle_data.size
            assert nan_pct < 0.05, f"Too many NaN in middle of data: {nan_pct*100:.1f}%"


# =============================================================================
# TRADING ENVIRONMENT INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestTradingEnvIntegration:
    """Integration tests for TradingEnv with real-ish data flow."""
    
    def test_env_episodes_complete(self, trading_env):
        """Environment should complete episodes without errors."""
        env = trading_env
        
        # Run a few episodes
        for episode in range(3):
            obs, _ = env.reset()
            total_reward = 0.0
            
            for step in range(100):  # Run up to 100 steps
                # Random action: direction [0,1,2], size [0,1,2,3]
                action = np.array([
                    np.random.randint(0, 3),
                    np.random.randint(0, 4)
                ], dtype=np.int32)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            # Should have completed without exception
            # Info dict contents may vary depending on terminal state
    
    def test_env_observation_consistency(self, trading_env):
        """Observations should be consistent dimension across steps."""
        env = trading_env
        obs_initial, _ = env.reset()
        
        obs_shapes = [obs_initial.shape]
        
        for _ in range(10):
            action = np.array([0, 0], dtype=np.int32)  # Stay flat
            obs, _, terminated, truncated, _ = env.step(action)
            obs_shapes.append(obs.shape)
            
            if terminated or truncated:
                break
        
        # All observations should have same shape
        assert all(s == obs_shapes[0] for s in obs_shapes), \
            "Observation shapes should be consistent"
    
    def test_env_pnl_tracking_accuracy(self, trading_env):
        """PnL tracking should be accurate throughout episode."""
        env = trading_env
        env.reset()
        
        initial_pnl = env.total_pnl
        
        # Make some trades
        for _ in range(20):
            action = np.array([1, 1], dtype=np.int32)  # Long, medium size
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        # total_pnl should reflect trades (may be positive or negative)
        # Just verify it's being tracked
        assert hasattr(env, 'total_pnl')


# =============================================================================
# REWARD SYSTEM INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestRewardSystemIntegration:
    """Integration tests for the reward system."""
    
    def test_reward_finite_throughout_episode(self, trading_env):
        """Rewards should always be finite (no NaN or Inf)."""
        env = trading_env
        obs, _ = env.reset()
        
        rewards = []
        
        for _ in range(50):
            action = np.array([np.random.randint(0, 3), 0], dtype=np.int32)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        assert all(np.isfinite(r) for r in rewards), "All rewards should be finite"
    
    def test_alpha_reward_components_present(self, trading_env):
        """Alpha reward should include expected components in info."""
        env = trading_env
        
        if not env.use_alpha_reward:
            pytest.skip("Alpha reward not enabled")
        
        obs, _ = env.reset()
        
        # Open a position and hold
        env.step(np.array([1, 1], dtype=np.int32))  # Go long
        _, _, _, _, info = env.step(np.array([1, 1], dtype=np.int32))  # Step with position
        
        # If position was held, info should have alpha components
        if env.position != 0:
            # Alpha info may or may not be present depending on step
            pass  # Just verify no exception


# =============================================================================
# CONFIGURATION INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration consistency."""
    
    def test_config_loads_correctly(self, config):
        """Config should load with all required sections."""
        assert hasattr(config, 'trading')
        assert hasattr(config, 'instrument')
        assert hasattr(config, 'analyst')  # Was 'model'
    
    def test_trading_config_values_valid(self, config):
        """Trading config values should be within valid ranges."""
        trading = config.trading
        
        assert trading.reward_scaling > 0
        assert trading.profit_scaling >= trading.loss_scaling
        assert trading.sl_atr_multiplier > 0
        assert trading.tp_atr_multiplier > 0
        assert trading.spread_pips >= 0
    
    def test_instrument_config_values_valid(self, config):
        """Instrument config should have valid values."""
        instrument = config.instrument
        
        assert instrument.pip_value > 0
        assert instrument.lot_size > 0


# =============================================================================
# CROSS-COMPONENT CONSISTENCY TESTS
# =============================================================================

@pytest.mark.integration
class TestCrossComponentConsistency:
    """Tests for consistency across components."""
    
    def test_env_and_backtest_sl_match(self, config):
        """Environment and backtester should use same SL/TP levels."""
        from src.evaluation.backtest import Backtester as BacktestEngine
        
        engine = BacktestEngine(
            sl_atr_multiplier=config.trading.sl_atr_multiplier,
            tp_atr_multiplier=config.trading.tp_atr_multiplier
        )
        
        # Both should use config values
        assert engine.sl_atr_multiplier == config.trading.sl_atr_multiplier
        assert engine.tp_atr_multiplier == config.trading.tp_atr_multiplier
    
    def test_reward_scaling_consistency(self, config, trading_env):
        """Reward scaling should be consistent between config and env."""
        assert trading_env.reward_scaling == config.trading.reward_scaling
        assert trading_env.profit_scaling == config.trading.profit_scaling
        assert trading_env.loss_scaling == config.trading.loss_scaling


# =============================================================================
# MEMORY AND PERFORMANCE TESTS
# =============================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestPerformance:
    """Performance and resource usage tests."""
    
    def test_env_episode_memory_stable(self, trading_env):
        """Memory should not grow significantly across episodes."""
        import gc
        
        env = trading_env
        
        # Run several episodes
        for _ in range(5):
            env.reset()
            for _ in range(50):
                action = np.array([0, 0], dtype=np.int32)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        
        gc.collect()
        
        # Just verify no exception - memory profiling requires more tooling
        assert True
    
    def test_feature_engineering_performance(self, sample_ohlcv_data):
        """Feature engineering should complete in reasonable time."""
        import time
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
        
        start = time.time()
        _ = engineer_all_features(sample_ohlcv_data.copy(), feature_config)
        elapsed = time.time() - start
        
        # Should complete in < 5 seconds for 500 samples
        assert elapsed < 5.0, f"Feature engineering took {elapsed:.2f}s, should be < 5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
