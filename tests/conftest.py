"""
Elite Test Suite - Shared Fixtures & Configuration
===================================================
Professional pytest fixtures for US30 AI Trading Bot.

Provides:
- Synthetic market data generators
- Pre-configured environment instances
- Feature DataFrame factories
- Parametrized market scenarios
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def config():
    """Load production configuration."""
    return Config()


@pytest.fixture(scope="session")
def device():
    """Get PyTorch device for testing."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# =============================================================================
# MARKET DATA GENERATORS
# =============================================================================

@dataclass
class MarketScenario:
    """Encapsulates a complete market scenario for testing."""
    name: str
    ohlcv: pd.DataFrame
    expected_direction: str  # 'bullish', 'bearish', 'sideways'
    expected_volatility: str  # 'low', 'medium', 'high'
    
    @property
    def close_prices(self) -> np.ndarray:
        return self.ohlcv['close'].values.astype(np.float32)


def _generate_ohlcv(
    n_bars: int,
    start_price: float = 20000.0,
    trend: float = 0.0,
    volatility: float = 50.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with controlled characteristics.
    
    Args:
        n_bars: Number of bars to generate
        start_price: Starting close price
        trend: Daily trend in points (positive = bullish)
        volatility: Standard deviation of price moves
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with OHLCV columns and DatetimeIndex
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate returns with trend
    returns = np.random.normal(trend, volatility, n_bars)
    
    # Cumulative price
    close = start_price + np.cumsum(returns)
    close = np.maximum(close, 1000)  # Floor at 1000
    
    # Generate OHLC from close
    high_pct = np.abs(np.random.normal(0.002, 0.001, n_bars))
    low_pct = np.abs(np.random.normal(0.002, 0.001, n_bars))
    open_pct = np.random.normal(0, 0.001, n_bars)
    
    high = close * (1 + high_pct)
    low = close * (1 - low_pct)
    open_ = close * (1 + open_pct)
    
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    
    # Volume
    volume = np.random.uniform(1000, 10000, n_bars)
    
    # Timestamps (5-minute bars)
    start_time = datetime(2024, 1, 1, 9, 0)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_bars)]
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=pd.DatetimeIndex(timestamps))
    
    return df.astype(np.float32)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Basic 500-bar OHLCV data for general testing."""
    return _generate_ohlcv(500, start_price=20000, trend=0, volatility=50, seed=42)


@pytest.fixture
def bullish_market_data() -> MarketScenario:
    """Strong uptrend scenario (+100 pts/bar average)."""
    ohlcv = _generate_ohlcv(500, start_price=20000, trend=10, volatility=30, seed=100)
    return MarketScenario(
        name="strong_bullish",
        ohlcv=ohlcv,
        expected_direction="bullish",
        expected_volatility="medium"
    )


@pytest.fixture
def bearish_market_data() -> MarketScenario:
    """Strong downtrend scenario (-100 pts/bar average)."""
    ohlcv = _generate_ohlcv(500, start_price=22000, trend=-10, volatility=30, seed=101)
    return MarketScenario(
        name="strong_bearish",
        ohlcv=ohlcv,
        expected_direction="bearish",
        expected_volatility="medium"
    )


@pytest.fixture
def sideways_market_data() -> MarketScenario:
    """Ranging/choppy market scenario."""
    ohlcv = _generate_ohlcv(500, start_price=20000, trend=0, volatility=80, seed=102)
    return MarketScenario(
        name="sideways_choppy",
        ohlcv=ohlcv,
        expected_direction="sideways",
        expected_volatility="high"
    )


@pytest.fixture
def low_volatility_data() -> MarketScenario:
    """Low volatility quiet market."""
    ohlcv = _generate_ohlcv(500, start_price=20000, trend=2, volatility=15, seed=103)
    return MarketScenario(
        name="low_vol_drift",
        ohlcv=ohlcv,
        expected_direction="bullish",
        expected_volatility="low"
    )


@pytest.fixture(params=["bullish", "bearish", "sideways"])
def all_market_scenarios(request, bullish_market_data, bearish_market_data, sideways_market_data):
    """Parametrized fixture that runs test across all market scenarios."""
    scenarios = {
        "bullish": bullish_market_data,
        "bearish": bearish_market_data,
        "sideways": sideways_market_data
    }
    return scenarios[request.param]


# =============================================================================
# FEATURE ENGINEERING FIXTURES
# =============================================================================

@pytest.fixture
def sample_features_df(sample_ohlcv_data) -> pd.DataFrame:
    """DataFrame with all engineered features for testing."""
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
    
    return engineer_all_features(sample_ohlcv_data.copy(), feature_config)


@pytest.fixture
def market_features_array(sample_features_df) -> np.ndarray:
    """Market features array matching TradingEnv format."""
    market_cols = ['atr', 'chop', 'adx', 'sma_distance']
    available_cols = [c for c in market_cols if c in sample_features_df.columns]
    
    # Fill NaN with reasonable defaults
    df = sample_features_df.copy()
    for col in available_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df[available_cols].values.astype(np.float32)


# =============================================================================
# TRADING ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def minimal_trading_env_data(sample_ohlcv_data, market_features_array):
    """Minimal data required to instantiate TradingEnv."""
    n_samples = min(len(sample_ohlcv_data), len(market_features_array))
    
    # Create dummy windowed data
    lookback = 48
    n_features = 10
    
    data_5m = np.random.randn(n_samples, lookback, n_features).astype(np.float32)
    data_15m = np.random.randn(n_samples, 16, n_features).astype(np.float32)
    data_45m = np.random.randn(n_samples, 6, n_features).astype(np.float32)
    
    close_prices = sample_ohlcv_data['close'].values[:n_samples].astype(np.float32)
    market_features = market_features_array[:n_samples]
    
    return {
        'data_5m': data_5m,
        'data_15m': data_15m,
        'data_45m': data_45m,
        'close_prices': close_prices,
        'market_features': market_features
    }


@pytest.fixture
def trading_env(minimal_trading_env_data, config, device):
    """Pre-configured TradingEnv instance for testing."""
    from src.environments.trading_env import TradingEnv
    
    env = TradingEnv(
        data_5m=minimal_trading_env_data['data_5m'],
        data_15m=minimal_trading_env_data['data_15m'],
        data_45m=minimal_trading_env_data['data_45m'],
        close_prices=minimal_trading_env_data['close_prices'],
        market_features=minimal_trading_env_data['market_features'],
        analyst_model=None,  # No analyst for unit tests
        device=device,
        # Use correct config paths - pip_value is in InstrumentConfig
        pip_value=config.instrument.pip_value,
        spread_pips=config.trading.spread_pips,
        reward_scaling=config.trading.reward_scaling,
        profit_scaling=config.trading.profit_scaling,
        loss_scaling=config.trading.loss_scaling,
        use_alpha_reward=config.trading.use_alpha_reward,
        alpha_baseline_exposure=config.trading.alpha_baseline_exposure,
        sl_atr_multiplier=config.trading.sl_atr_multiplier,
        tp_atr_multiplier=config.trading.tp_atr_multiplier,
        # v36: Include min_hold_bars from config to test scalping prevention
        min_hold_bars=config.trading.min_hold_bars,
        max_steps=100  # Shorter for faster tests
    )
    
    return env


# =============================================================================
# HELPER FUNCTIONS FOR TESTS
# =============================================================================

def assert_no_lookahead(series: pd.Series, lookback: int = 0):
    """
    Assert that a series has no look-ahead bias.
    
    Values at index i should only use data from indices <= i - lookback.
    """
    # Check that NaN exists at the start (proper warmup period)
    n_initial_nan = series.isna().sum()
    assert n_initial_nan >= lookback, \
        f"Expected at least {lookback} initial NaN values, got {n_initial_nan}"


def assert_within_tolerance(actual: float, expected: float, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert two values are approximately equal."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_pnl_calculation_correct(
    entry_price: float,
    exit_price: float,
    position: int,
    position_size: float,
    pip_value: float,
    expected_pnl: float
):
    """Assert PnL calculation is correct."""
    if position == 1:  # Long
        pnl_pips = (exit_price - entry_price) / pip_value
    else:  # Short
        pnl_pips = (entry_price - exit_price) / pip_value
    
    actual_pnl = pnl_pips * position_size
    assert_within_tolerance(actual_pnl, expected_pnl, rtol=1e-4)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "parametrize_market: runs test across multiple market scenarios"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test names."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
