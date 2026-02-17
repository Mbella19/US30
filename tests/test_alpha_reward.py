"""
Elite Test Suite - Alpha Reward Calculation Tests
==================================================
Comprehensive verification of alpha-based reward mechanism.

Tests verify:
- Position-specific baseline comparison (Long vs B&H, Short vs S&H)
- Asymmetric scaling (1.5x profit, 1.0x loss)
- Spread cost deduction per step
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
from typing import Tuple

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ALPHA CALCULATION HELPER (Mirrors TradingEnv Logic)
# =============================================================================

def calculate_alpha(
    pnl_delta: float,
    market_move_pips: float,
    position: int,
    position_size: float,
    alpha_baseline_exposure: float = 1.0
) -> float:
    """
    Calculate alpha (outperformance vs baseline strategy).
    
    Args:
        pnl_delta: Agent's PnL change this step
        market_move_pips: Market price change in pips
        position: 1 (long), -1 (short), 0 (flat)
        position_size: Size of position
        alpha_baseline_exposure: Baseline exposure factor
        
    Returns:
        Alpha value (positive = outperformed)
    """
    if position == 1:  # Long
        # Compare to buy-and-hold
        baseline_pnl = market_move_pips * alpha_baseline_exposure * position_size
    elif position == -1:  # Short
        # Compare to sell-and-hold
        baseline_pnl = -market_move_pips * alpha_baseline_exposure * position_size
    else:
        baseline_pnl = 0
    
    return pnl_delta - baseline_pnl


def calculate_alpha_reward(
    alpha: float,
    profit_scaling: float = 0.015,
    loss_scaling: float = 0.01
) -> float:
    """Apply asymmetric scaling to alpha."""
    if alpha > 0:
        return alpha * profit_scaling
    else:
        return alpha * loss_scaling


def calculate_spread_cost_per_step(
    spread_pips: float,
    reward_scaling: float,
    position_size: float,
    expected_hold_bars: int = 20
) -> float:
    """Calculate amortized spread cost per step."""
    return (spread_pips * reward_scaling * position_size) / expected_hold_bars


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def alpha_params():
    """Default alpha reward parameters from config."""
    return {
        'profit_scaling': 0.015,
        'loss_scaling': 0.01,
        'alpha_baseline_exposure': 0.5,  # v32 uses 50%
        'spread_pips': 50.0,
        'reward_scaling': 0.01
    }


# =============================================================================
# LONG POSITION ALPHA TESTS
# =============================================================================

class TestLongPositionAlpha:
    """Tests for alpha calculation when holding a long position."""
    
    def test_long_in_bull_with_full_exposure_alpha_zero(self):
        """
        Long position in bull market with 100% baseline exposure.
        Should have ZERO alpha (matched buy-and-hold exactly).
        """
        market_move = 100.0  # Market up 100 pips
        pnl_delta = 100.0    # Long gained 100 pips
        position = 1
        position_size = 1.0
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=position,
            position_size=position_size,
            alpha_baseline_exposure=1.0  # 100% exposure
        )
        
        assert alpha == 0.0, f"Long matching market should have 0 alpha, got {alpha}"
    
    def test_long_in_bull_with_half_exposure_positive_alpha(self):
        """
        Long position in bull market with 50% baseline exposure.
        Should have POSITIVE alpha (beat the baseline).
        """
        market_move = 100.0
        pnl_delta = 100.0
        position = 1
        position_size = 1.0
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=position,
            position_size=position_size,
            alpha_baseline_exposure=0.5  # 50% exposure
        )
        
        # baseline = 100 * 0.5 * 1.0 = 50
        # alpha = 100 - 50 = 50
        assert alpha == 50.0, f"Expected alpha=50, got {alpha}"
    
    def test_long_in_bear_with_full_exposure_alpha_zero(self):
        """
        Long position in bear market with 100% baseline exposure.
        Should have ZERO alpha (matched buy-and-hold loss exactly).
        """
        market_move = -100.0  # Market down 100 pips
        pnl_delta = -100.0    # Long lost 100 pips
        position = 1
        position_size = 1.0
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=position,
            position_size=position_size,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == 0.0, f"Long matching market loss should have 0 alpha, got {alpha}"
    
    def test_long_outperforms_market(self):
        """Long gains more than market move = positive alpha."""
        market_move = 100.0
        pnl_delta = 150.0  # Gained more than market
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=1,
            position_size=1.0,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == 50.0, f"Outperformance should give +50 alpha, got {alpha}"
    
    def test_long_underperforms_market(self):
        """Long gains less than market move = negative alpha."""
        market_move = 100.0
        pnl_delta = 50.0  # Gained less (perhaps due to slippage)
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=1,
            position_size=1.0,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == -50.0, f"Underperformance should give -50 alpha, got {alpha}"


# =============================================================================
# SHORT POSITION ALPHA TESTS
# =============================================================================

class TestShortPositionAlpha:
    """Tests for alpha calculation when holding a short position."""
    
    def test_short_in_bear_with_full_exposure_alpha_zero(self):
        """
        Short position in bear market with 100% baseline exposure.
        Should have ZERO alpha (matched sell-and-hold exactly).
        """
        market_move = -100.0  # Market down 100 pips
        pnl_delta = 100.0     # Short gained 100 pips
        position = -1
        position_size = 1.0
        
        # baseline for short = -market_move * exposure * size = -(-100) * 1.0 * 1.0 = 100
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=position,
            position_size=position_size,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == 0.0, f"Short matching S&H should have 0 alpha, got {alpha}"
    
    def test_short_in_bull_with_full_exposure_alpha_zero(self):
        """
        Short position in bull market with 100% baseline exposure.
        Unlike buy-and-hold comparison, sell-and-hold also loses in bull.
        Should have ZERO alpha (matched sell-and-hold loss).
        """
        market_move = 100.0   # Market up 100 pips
        pnl_delta = -100.0    # Short lost 100 pips
        position = -1
        position_size = 1.0
        
        # baseline for short = -market_move * exposure * size = -(100) * 1.0 * 1.0 = -100
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=position,
            position_size=position_size,
            alpha_baseline_exposure=1.0
        )
        
        # alpha = -100 - (-100) = 0
        assert alpha == 0.0, f"Short matching S&H loss should have 0 alpha, got {alpha}"
    
    def test_short_in_bull_with_half_exposure_negative_alpha(self):
        """
        Short position in bull market with 50% baseline exposure.
        Should have NEGATIVE alpha vs 50% S&H baseline.
        """
        market_move = 100.0
        pnl_delta = -100.0  # Lost 100
        position = -1
        position_size = 1.0
        
        # baseline = -100 * 0.5 * 1.0 = -50 (S&H would lose 50)
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=position,
            position_size=position_size,
            alpha_baseline_exposure=0.5
        )
        
        # alpha = -100 - (-50) = -50
        assert alpha == -50.0, f"Expected alpha=-50, got {alpha}"
    
    def test_short_outperforms_sell_and_hold(self):
        """Short gains more than S&H baseline = positive alpha."""
        market_move = -100.0
        pnl_delta = 150.0  # Gained 150 (better than 100)
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=-1,
            position_size=1.0,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == 50.0, f"Short outperformance should give +50 alpha, got {alpha}"


# =============================================================================
# ASYMMETRIC SCALING TESTS
# =============================================================================

class TestAsymmetricScaling:
    """Tests for asymmetric profit/loss scaling."""
    
    def test_positive_alpha_gets_profit_scaling(self):
        """Positive alpha should use profit_scaling (1.5x)."""
        alpha = 100.0
        profit_scaling = 0.015
        loss_scaling = 0.01
        
        reward = calculate_alpha_reward(alpha, profit_scaling, loss_scaling)
        expected = 100.0 * 0.015
        
        assert reward == expected, f"Expected {expected}, got {reward}"
    
    def test_negative_alpha_gets_loss_scaling(self):
        """Negative alpha should use loss_scaling (1.0x)."""
        alpha = -100.0
        profit_scaling = 0.015
        loss_scaling = 0.01
        
        reward = calculate_alpha_reward(alpha, profit_scaling, loss_scaling)
        expected = -100.0 * 0.01
        
        assert reward == expected, f"Expected {expected}, got {reward}"
    
    def test_zero_alpha_gives_zero_reward(self):
        """Zero alpha should give zero reward."""
        alpha = 0.0
        
        reward = calculate_alpha_reward(alpha, 0.015, 0.01)
        
        assert reward == 0.0
    
    def test_asymmetric_creates_positive_ev(self):
        """
        With equal +100 and -100 alpha, asymmetric scaling creates positive EV.
        This is by design to encourage exploration.
        """
        profit_scaling = 0.015
        loss_scaling = 0.01
        
        reward_win = calculate_alpha_reward(100.0, profit_scaling, loss_scaling)
        reward_loss = calculate_alpha_reward(-100.0, profit_scaling, loss_scaling)
        
        # EV = 0.5 * (+1.5) + 0.5 * (-1.0) = +0.25
        net = reward_win + reward_loss
        
        assert net > 0, f"Asymmetric scaling should create positive EV, got {net}"
        assert net == pytest.approx(0.5, rel=1e-4), f"Expected 0.5, got {net}"


# =============================================================================
# SPREAD COST DEDUCTION TESTS
# =============================================================================

class TestSpreadCostDeduction:
    """Tests for per-step spread cost deduction."""
    
    def test_spread_cost_calculation(self):
        """Verify spread cost per step calculation."""
        spread_pips = 50.0
        reward_scaling = 0.01
        position_size = 1.0
        expected_hold_bars = 20
        
        cost = calculate_spread_cost_per_step(
            spread_pips, reward_scaling, position_size, expected_hold_bars
        )
        
        # (50 * 0.01 * 1.0) / 20 = 0.025
        assert cost == pytest.approx(0.025, rel=1e-6)
    
    def test_spread_cost_scales_with_position_size(self):
        """Larger position = larger spread cost."""
        cost_small = calculate_spread_cost_per_step(50.0, 0.01, 0.5, 20)
        cost_large = calculate_spread_cost_per_step(50.0, 0.01, 2.0, 20)
        
        assert cost_large == 4 * cost_small
    
    def test_matching_market_with_spread_gives_negative_reward(self):
        """
        Agent matching market exactly should get NEGATIVE reward
        because spread cost is deducted.
        """
        alpha = 0.0  # Matched market
        alpha_reward = calculate_alpha_reward(alpha, 0.015, 0.01)  # 0
        spread_cost = calculate_spread_cost_per_step(50.0, 0.01, 1.0, 20)  # 0.025
        
        net_reward = alpha_reward - spread_cost
        
        assert net_reward < 0, f"Matching market should give negative reward, got {net_reward}"
        assert net_reward == pytest.approx(-0.025, rel=1e-6)
    
    def test_total_spread_over_hold_equals_entry_cost(self):
        """Sum of per-step costs over hold period = full spread cost."""
        spread_pips = 50.0
        reward_scaling = 0.01
        position_size = 1.0
        hold_bars = 20
        
        cost_per_step = calculate_spread_cost_per_step(
            spread_pips, reward_scaling, position_size, hold_bars
        )
        total_cost = cost_per_step * hold_bars
        
        expected_full_cost = spread_pips * reward_scaling * position_size
        
        assert total_cost == pytest.approx(expected_full_cost, rel=1e-6)


# =============================================================================
# EDGE CASES & BOUNDARY CONDITIONS
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_zero_market_move_zero_alpha(self):
        """No market move = zero baseline = alpha equals pnl_delta."""
        pnl_delta = 50.0  # Maybe from position sizing adjustment
        market_move = 0.0
        
        alpha = calculate_alpha(
            pnl_delta=pnl_delta,
            market_move_pips=market_move,
            position=1,
            position_size=1.0,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == pnl_delta
    
    def test_flat_position_zero_baseline(self):
        """Flat position should always have zero baseline PnL."""
        alpha = calculate_alpha(
            pnl_delta=0.0,
            market_move_pips=100.0,
            position=0,
            position_size=0.0,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == 0.0
    
    def test_very_small_alpha_still_scaled(self):
        """Even tiny alpha values should be scaled correctly."""
        alpha = 0.001
        reward = calculate_alpha_reward(alpha, 0.015, 0.01)
        
        assert reward == pytest.approx(0.001 * 0.015, rel=1e-6)
    
    def test_large_position_size_scales_baseline(self):
        """Larger position size increases baseline PnL proportionally."""
        market_move = 100.0
        pnl_delta_small = 50.0   # 0.5 lots * 100 pips
        pnl_delta_large = 200.0  # 2.0 lots * 100 pips
        
        alpha_small = calculate_alpha(
            pnl_delta=pnl_delta_small,
            market_move_pips=market_move,
            position=1,
            position_size=0.5,
            alpha_baseline_exposure=1.0
        )
        
        alpha_large = calculate_alpha(
            pnl_delta=pnl_delta_large,
            market_move_pips=market_move,
            position=1,
            position_size=2.0,
            alpha_baseline_exposure=1.0
        )
        
        # Both matched market: alpha = 0
        assert alpha_small == 0.0
        assert alpha_large == 0.0


# =============================================================================
# PARAMETRIZED COMPREHENSIVE TESTS
# =============================================================================

class TestParametrizedAlphaScenarios:
    """Parametrized tests covering many scenarios."""
    
    @pytest.mark.parametrize("position,market_move,pnl_delta,expected_alpha", [
        # Long in various markets
        (1, 100, 100, 0),      # Long matches bull
        (1, -100, -100, 0),    # Long matches bear
        (1, 100, 150, 50),     # Long outperforms
        (1, 100, 50, -50),     # Long underperforms
        # Short in various markets (with S&H baseline)
        (-1, -100, 100, 0),    # Short matches bear
        (-1, 100, -100, 0),    # Short matches bull loss
        (-1, -100, 150, 50),   # Short outperforms
        (-1, -100, 50, -50),   # Short underperforms
    ])
    def test_alpha_scenarios_full_exposure(
        self, position, market_move, pnl_delta, expected_alpha
    ):
        """Test alpha calculation across various scenarios with 100% exposure."""
        alpha = calculate_alpha(
            pnl_delta=float(pnl_delta),
            market_move_pips=float(market_move),
            position=position,
            position_size=1.0,
            alpha_baseline_exposure=1.0
        )
        
        assert alpha == pytest.approx(expected_alpha, rel=1e-6), \
            f"Position={position}, move={market_move}, pnl={pnl_delta}: expected {expected_alpha}, got {alpha}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
