"""
Feature engineering module for the hybrid trading system.

Implements:
- Market Structure: TV-style S/R Channels, v2 Structure Breaks (BOS/CHoCH)
- Trend Filters: SMA distance
- Regime Detection: Choppiness Index, ADX
- Mean Reversion: Bollinger Bands, RSI, Williams %R, CCI, RSI Divergence
- OOD Detection: Training-anchored and rolling-window features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Market Structure
# =============================================================================

def detect_fractals(
    df: pd.DataFrame,
    n: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Williams Fractals for Support/Resistance levels.

    IMPORTANT: Uses DELAYED detection to prevent look-ahead bias!
    A fractal is only marked AFTER it's confirmed (when we have n bars after it).

    At time T, we can only know about fractals at time T-n or earlier,
    because we need n bars AFTER the fractal point to confirm it.

    A fractal high at bar i requires:
    - high[i] > high[i-1], high[i-2] (past - OK)
    - high[i] > high[i+1], high[i+2] (future - must wait for these)

    So at bar i+2, we can finally confirm the fractal at bar i.
    We mark the fractal at i+2 (current bar) with the VALUE from bar i.

    Args:
        df: OHLCV DataFrame
        n: Total window size (must be odd, e.g., 5 means 2 bars each side)

    Returns:
        Tuple of (fractal_highs, fractal_lows) as boolean Series
        NOTE: These are DELAYED - the fractal occurred n//2 bars AGO
    """
    half_n = n // 2

    fractal_highs = pd.Series(False, index=df.index)
    fractal_lows = pd.Series(False, index=df.index)

    # Start from position where we have enough PAST data to confirm a fractal
    # At position i, we're checking if position (i - half_n) was a fractal
    # This means we're only using data from [i - n + 1] to [i] (all past/current)
    for i in range(n - 1, len(df)):
        # The candidate fractal point is half_n bars AGO
        fractal_idx = i - half_n

        # Window is [fractal_idx - half_n, fractal_idx + half_n] = [i - n + 1, i]
        # All of this is past data relative to current position i
        window_start = fractal_idx - half_n
        window_end = fractal_idx + half_n + 1  # +1 for slice

        window_high = df['high'].iloc[window_start:window_end]
        window_low = df['low'].iloc[window_start:window_end]

        # Check if the candidate point (half_n bars ago) is a fractal
        candidate_high = df['high'].iloc[fractal_idx]
        candidate_low = df['low'].iloc[fractal_idx]

        # Mark at CURRENT position (i), indicating we NOW KNOW about this fractal
        # The actual S/R level is at df['high'].iloc[fractal_idx]
        # FIXED: Use tolerance-based comparison to handle float precision issues
        if (abs(candidate_high - window_high.max()) < 1e-10 and 
            candidate_high > window_high.iloc[0] and 
            candidate_high > window_high.iloc[-1]):
            fractal_highs.iloc[i] = True

        if (abs(candidate_low - window_low.min()) < 1e-10 and 
            candidate_low < window_low.iloc[0] and 
            candidate_low < window_low.iloc[-1]):
            fractal_lows.iloc[i] = True

    return fractal_highs, fractal_lows


def get_fractal_levels(
    df: pd.DataFrame,
    fractal_window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Get the actual price levels of fractals (with delayed detection).

    Since fractals are detected with a delay, at position i where
    fractal_highs[i] = True, the actual fractal price is from
    position i - (fractal_window // 2).

    Returns:
        Tuple of (resistance_prices, support_prices) as Series
        NaN where no fractal was detected
    """
    half_n = fractal_window // 2
    fractal_highs, fractal_lows = detect_fractals(df, fractal_window)

    # Get the actual prices where fractals occurred
    resistance_prices = pd.Series(np.nan, index=df.index, dtype=np.float32)
    support_prices = pd.Series(np.nan, index=df.index, dtype=np.float32)

    for i in range(len(df)):
        if fractal_highs.iloc[i]:
            # The actual fractal was half_n bars ago
            actual_fractal_idx = i - half_n
            if actual_fractal_idx >= 0:
                resistance_prices.iloc[i] = df['high'].iloc[actual_fractal_idx]

        if fractal_lows.iloc[i]:
            actual_fractal_idx = i - half_n
            if actual_fractal_idx >= 0:
                support_prices.iloc[i] = df['low'].iloc[actual_fractal_idx]

    return resistance_prices, support_prices


def get_sr_levels(
    df: pd.DataFrame,
    fractal_window: int = 5,
    lookback: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Get current Support and Resistance levels from recent fractals.

    Uses delayed fractal detection (no look-ahead bias).

    Returns:
        Tuple of (resistance_levels, support_levels)
    """
    resistance_prices, support_prices = get_fractal_levels(df.tail(lookback + fractal_window), fractal_window)

    resistance = resistance_prices.dropna().tolist()
    support = support_prices.dropna().tolist()

    return resistance, support


def distance_to_nearest_sr(
    price: pd.Series,
    df: pd.DataFrame,
    atr: pd.Series,
    fractal_window: int = 21,
    lookback: int = 290,
    max_channels: int = 6,
    channel_width_pct: float = 4.5,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate ATR-normalized distance to nearest S/R channels with strength.

    Inspired by TradingView's "Support Resistance Channels" (LonesomeTheBlue):
    1. Detect pivots (fractal highs + lows pooled together)
    2. Build channels by absorbing nearby pivots within adaptive width
    3. Score channels: pivot count * 20 + bar touch count
    4. Greedily select top-N non-overlapping channels by strength
    5. Classify as resistance (above price) or support (below price)

    Channel width adapts to volatility: 4.5% of the 300-bar price range.

    Uses delayed fractal detection (no look-ahead bias).

    Args:
        price: Close price series
        df: OHLCV DataFrame
        atr: ATR series
        fractal_window: Fractal detection window (21 = ~1.75h swings on 5m)
        lookback: Bars to look back for pivots (290 = ~24h on 5m)
        max_channels: Maximum S/R channels to keep (6 = trader standard)
        channel_width_pct: Channel width as % of 300-bar price range

    Returns:
        Tuple of (dist_to_resistance, dist_to_support,
                  sr_strength_r, sr_strength_s)
        - dist_to_*: ATR-normalized distance to nearest channel edge
        - sr_strength_*: Normalized strength (0-1) of nearest channel
    """
    length = len(price)
    half_n = fractal_window // 2

    dist_to_r = np.zeros(length, dtype=np.float32)
    dist_to_s = np.zeros(length, dtype=np.float32)
    strength_r = np.zeros(length, dtype=np.float32)
    strength_s = np.zeros(length, dtype=np.float32)

    # Pre-extract arrays
    price_arr = price.values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = atr.values

    # Detect fractals — pool highs and lows together as "pivots"
    fractal_highs, fractal_lows = detect_fractals(df, fractal_window)
    fh_arr = fractal_highs.values
    fl_arr = fractal_lows.values

    # Build sorted pivot list: (detection_bar, price_level)
    pivot_det_bars = []   # bar where pivot was detected (confirmed)
    pivot_prices = []     # actual price level of the pivot
    for i in range(length):
        if fh_arr[i]:
            idx = i - half_n
            if idx >= 0:
                pivot_det_bars.append(i)
                pivot_prices.append(high_arr[idx])
        if fl_arr[i]:
            idx = i - half_n
            if idx >= 0:
                pivot_det_bars.append(i)
                pivot_prices.append(low_arr[idx])

    pivot_det_bars = np.array(pivot_det_bars, dtype=np.int64)
    pivot_prices = np.array(pivot_prices, dtype=np.float64)

    # Cache: only rebuild channels when pivot set changes
    cached_channels = []  # list of (hi, lo, strength)
    cached_max_strength = 1.0
    last_active_count = -1
    last_oldest_pivot = -1

    # Pointer for binary search into sorted pivot_det_bars
    for i in range(max(lookback, 300), length):
        current_price = price_arr[i]
        current_atr = atr_arr[i]

        if current_atr <= 0 or np.isnan(current_atr) or np.isnan(current_price):
            continue

        # Active pivots: detected in [i-lookback, i)
        window_start = i - lookback
        active_mask = (pivot_det_bars >= window_start) & (pivot_det_bars < i)
        active_prices = pivot_prices[active_mask]
        active_count = len(active_prices)

        # Determine oldest active pivot for cache invalidation
        oldest = int(pivot_det_bars[active_mask][0]) if active_count > 0 else -1

        # Rebuild channels only when pivot set changes
        if active_count != last_active_count or oldest != last_oldest_pivot:
            last_active_count = active_count
            last_oldest_pivot = oldest

            if active_count == 0:
                cached_channels = []
                cached_max_strength = 1.0
                continue

            # Adaptive channel width: channel_width_pct% of 300-bar range
            range_start = max(0, i - 300)
            range_high = np.max(high_arr[range_start:i])
            range_low = np.min(low_arr[range_start:i])
            cwidth = (range_high - range_low) * channel_width_pct / 100.0

            if cwidth < current_atr * 0.1:
                cwidth = current_atr  # floor at 1 ATR

            # --- Build a channel from each pivot (TV algorithm) ---
            n_piv = len(active_prices)
            channel_his = np.empty(n_piv, dtype=np.float64)
            channel_los = np.empty(n_piv, dtype=np.float64)
            channel_strengths = np.zeros(n_piv, dtype=np.float64)

            for p in range(n_piv):
                lo = active_prices[p]
                hi = lo
                num_pivots = 0
                for q in range(n_piv):
                    cpp = active_prices[q]
                    wdth = (hi - cpp) if cpp <= hi else (cpp - lo)
                    if wdth <= cwidth:
                        lo = min(lo, cpp)
                        hi = max(hi, cpp)
                        num_pivots += 1
                channel_his[p] = hi
                channel_los[p] = lo
                channel_strengths[p] = num_pivots * 20

            # Add bar-touch count to strength
            bar_start = max(0, i - lookback)
            h_slice = high_arr[bar_start:i]
            l_slice = low_arr[bar_start:i]
            for p in range(n_piv):
                hi = channel_his[p]
                lo = channel_los[p]
                touches = np.sum(
                    ((h_slice >= lo) & (h_slice <= hi))
                    | ((l_slice >= lo) & (l_slice <= hi))
                )
                channel_strengths[p] += int(touches)

            # --- Greedy top-N non-overlapping selection ---
            order = np.argsort(-channel_strengths)  # strongest first
            selected = []
            for idx in order:
                s = channel_strengths[idx]
                if s < 20:  # minimum 1 pivot
                    continue
                hi = channel_his[idx]
                lo = channel_los[idx]
                # Check overlap with already selected
                overlaps = False
                for sh, sl, _ in selected:
                    if not (hi < sl or lo > sh):
                        overlaps = True
                        break
                if not overlaps:
                    selected.append((hi, lo, s))
                    if len(selected) >= max_channels:
                        break

            cached_channels = selected
            cached_max_strength = max((s for _, _, s in selected), default=1.0)

        # --- Compute distances to cached channels ---
        nearest_r_dist = np.inf
        nearest_r_str = 0.0
        nearest_s_dist = np.inf
        nearest_s_str = 0.0

        for hi, lo, s in cached_channels:
            if lo > current_price:
                # Entire channel above price → resistance
                d = (lo - current_price) / current_atr
                if d < nearest_r_dist:
                    nearest_r_dist = d
                    nearest_r_str = s
            elif hi < current_price:
                # Entire channel below price → support
                d = (current_price - hi) / current_atr
                if d < nearest_s_dist:
                    nearest_s_dist = d
                    nearest_s_str = s
            else:
                # Price is INSIDE the channel — it's both S and R
                d_up = (hi - current_price) / current_atr
                d_down = (current_price - lo) / current_atr
                if d_up < nearest_r_dist:
                    nearest_r_dist = d_up
                    nearest_r_str = s
                if d_down < nearest_s_dist:
                    nearest_s_dist = d_down
                    nearest_s_str = s

        if nearest_r_dist < np.inf:
            dist_to_r[i] = nearest_r_dist
            strength_r[i] = nearest_r_str / cached_max_strength
        if nearest_s_dist < np.inf:
            dist_to_s[i] = nearest_s_dist
            strength_s[i] = nearest_s_str / cached_max_strength

    return (
        pd.Series(dist_to_r, index=price.index, dtype=np.float32),
        pd.Series(dist_to_s, index=price.index, dtype=np.float32),
        pd.Series(strength_r, index=price.index, dtype=np.float32),
        pd.Series(strength_s, index=price.index, dtype=np.float32),
    )


# =============================================================================
# Trend Filters
# =============================================================================

def sma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """Simple Moving Average."""
    return df['close'].rolling(window=period).mean().astype(np.float32)


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return df['close'].ewm(span=period, adjust=False).mean().astype(np.float32)


def sma_distance(
    df: pd.DataFrame,
    atr: pd.Series,
    period: int = 50
) -> pd.Series:
    """
    Calculate ATR-normalized distance from SMA(period).

    Positive = price above SMA (bullish)
    Negative = price below SMA (bearish)
    
    FIXED: Added clipping to prevent extreme values when ATR is near zero.
    """
    sma_val = sma(df, period)
    distance = (df['close'] - sma_val) / atr.replace(0, 1e-10)
    # Clip to prevent extreme values
    distance = distance.clip(-100, 100)
    return distance.astype(np.float32)


def ema_gap(
    df: pd.DataFrame,
    atr_series: pd.Series,
    fast: int = 14,
    slow: int = 50
) -> pd.Series:
    """
    ATR-normalized EMA gap — continuous trend momentum strength.

    Measures the spread between fast and slow EMA, normalized by ATR
    for regime robustness. Different from sma_distance (price vs SMA):
    this measures the spread between two averages (momentum confirmation).

    Returns: Continuous float. Positive = bullish momentum, negative = bearish.
    """
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)
    gap = (ema_fast - ema_slow) / atr_series.replace(0, 1e-10)
    return gap.clip(-20, 20).astype(np.float32)


def ema_acceleration(
    df: pd.DataFrame,
    atr_series: pd.Series,
    fast: int = 14,
    slow: int = 50,
    lookback: int = 12
) -> pd.Series:
    """
    EMA acceleration — rate of change of the EMA gap.

    Measures whether trend momentum is expanding or contracting.
    Positive = momentum strengthening, negative = momentum fading.
    Critical for mean reversion: fading momentum = safer to fade.

    Args:
        lookback: Bars to measure change over (~1 hour on 5m)
    """
    gap = ema_gap(df, atr_series, fast, slow)
    accel = gap - gap.shift(lookback)
    return accel.clip(-10, 10).fillna(0).astype(np.float32)


def ema_crossover_recency(
    df: pd.DataFrame,
    fast: int = 14,
    slow: int = 50,
    half_life: int = 12
) -> pd.Series:
    """
    Decaying EMA crossover signal — fixes sparsity of binary crossover.

    Instead of 1/-1 only on the exact cross bar, applies exponential decay
    so recent crosses have strong signal, older crosses fade out.
    Matches the design philosophy of structure_fade (half-life decay).

    Args:
        half_life: Decay half-life in bars (~1 hour on 5m)

    Returns: Continuous float in [-1, 1]. Positive = recent bullish cross,
             negative = recent bearish cross, decays toward 0.
    """
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)

    # Detect crossover bars
    above_now = ema_fast > ema_slow
    above_prev = ema_fast.shift(1) > ema_slow.shift(1)

    bullish_cross = above_now & ~above_prev
    bearish_cross = ~above_now & above_prev

    # Build decaying signal
    decay = 0.5 ** (1.0 / half_life)
    result = np.zeros(len(df), dtype=np.float32)

    for i in range(1, len(df)):
        if bullish_cross.iloc[i]:
            result[i] = 1.0
        elif bearish_cross.iloc[i]:
            result[i] = -1.0
        else:
            result[i] = result[i - 1] * decay

    return pd.Series(result, index=df.index, dtype=np.float32)


# =============================================================================
# Regime Detection
# =============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()

    return atr_val.astype(np.float32)


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP).

    Values > 61.8 indicate ranging/choppy market.
    Values < 38.2 indicate trending market.
    
    FIXED: Now uses raw True Range instead of averaged ATR.
    """
    # Calculate raw True Range (not averaged)
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    tr_sum = true_range.rolling(window=period).sum()

    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()

    # Prevent division by zero and log of zero
    range_diff = (high_max - low_min).replace(0, 1e-10)
    ratio = tr_sum / range_diff
    # CRITICAL FIX: clip to 1.0 (not 1e-10) to prevent negative CHOP values
    # When ratio < 1, log10(ratio) < 0, producing invalid negative CHOP
    # CHOP should always be 0-100, so ratio must be >= 1.0
    ratio = ratio.clip(lower=1.0)

    chop = 100 * np.log10(ratio) / np.log10(period)

    return chop.astype(np.float32)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).

    Values > 25 indicate trending market.
    Values < 20 indicate ranging market.
    
    FIXED: Corrected pandas assignment bug for +DM/-DM calculation.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # FIXED: Use np.where for proper conditional assignment
    plus_dm_values = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )
    minus_dm_values = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )
    
    plus_dm = pd.Series(plus_dm_values, index=df.index, dtype=np.float32)
    minus_dm = pd.Series(minus_dm_values, index=df.index, dtype=np.float32)

    # True Range
    atr_val = atr(df, period)

    # Smoothed +DI and -DI using Wilder's smoothing (alpha = 1/period)
    # Standard ADX uses Wilder's EMA, NOT span-based EMA
    # Wilder's: alpha = 1/N ≈ 0.071 for N=14
    # Span-based: alpha = 2/(N+1) ≈ 0.133 for N=14 (too reactive)
    # Use reasonable ATR floor to avoid extreme DI values when ATR is near zero
    atr_safe = atr_val.clip(lower=1e-6)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_safe)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_safe)

    # DX and ADX (also using Wilder's smoothing)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx_val.astype(np.float32)


def market_regime(
    df: pd.DataFrame,
    chop_period: int = 14,
    adx_period: int = 14
) -> pd.Series:
    """
    Classify market regime based on CHOP and ADX.

    Returns:
        1: Trending (tradeable)
        0: Neutral
        -1: Ranging/Choppy (avoid)
    """
    chop = choppiness_index(df, chop_period)
    adx_val = adx(df, adx_period)

    result = pd.Series(0, index=df.index, dtype=np.float32)

    # Trending: low chop + high ADX (both conditions)
    trending = (chop < 38.2) & (adx_val > 25)
    # Ranging: high chop + low ADX
    ranging = (chop > 61.8) & (adx_val < 20)

    result[trending] = 1
    result[ranging] = -1

    return result


def detect_market_regime_direction(
    df: pd.DataFrame,
    lookback: int = 20,
    trend_threshold: float = 0.10,
    chop_period: int = 14,
    adx_period: int = 14
) -> pd.Series:
    """
    Classify market regime into BULLISH, BEARISH, or RANGING.

    This is crucial for regime-balanced training to prevent directional bias.
    If training data is predominantly bearish, the agent learns to short.
    By balancing across regimes, we ensure the agent learns both directions.

    Args:
        df: OHLCV DataFrame
        lookback: Period for trend calculation
        trend_threshold: Threshold for trend classification (in ATR units)
        chop_period: Period for choppiness calculation
        adx_period: Period for ADX calculation

    Returns:
        Series with values:
            1: BULLISH (uptrend)
            0: RANGING (sideways/choppy)
           -1: BEARISH (downtrend)
    """
    # Calculate trend direction using price change over lookback
    price_change = df['close'].diff(lookback)

    # Calculate ATR for normalization
    atr_val = atr(df, lookback)
    atr_val = atr_val.replace(0, 1e-10)  # Avoid division by zero

    # Normalized price change (in ATR units)
    normalized_change = price_change / (atr_val * lookback)

    # Get regime indicators
    chop = choppiness_index(df, chop_period)
    adx_val = adx(df, adx_period)

    # Initialize as ranging
    result = pd.Series(0, index=df.index, dtype=np.float32)

    # Bullish: price went up significantly AND not too choppy
    bullish = (normalized_change > trend_threshold) & (chop < 55)

    # Bearish: price went down significantly AND not too choppy
    bearish = (normalized_change < -trend_threshold) & (chop < 55)

    # If very choppy (CHOP > 60), force to ranging regardless of direction
    very_choppy = chop > 60

    result[bullish] = 1
    result[bearish] = -1
    result[very_choppy] = 0  # Override to ranging if very choppy

    return result


def compute_regime_labels(
    df: pd.DataFrame,
    lookback: int = 50,
    forward_window: int = 200  # IGNORED - kept for API compatibility
) -> np.ndarray:
    """
    Compute regime labels for regime-balanced sampling using BACKWARD-LOOKING data only.

    v35 FIX: Uses ONLY HISTORICAL data to prevent look-ahead bias.
    The forward_window parameter is ignored but kept for API compatibility.

    This fixes the critical generalization failure where agents trained with
    forward-looking regime labels performed well on OOS within the training
    period but failed on truly held-out future data.

    Regime Classification (backward-looking):
    - BULLISH (0): Price has risen significantly over lookback period
    - RANGING (1): Price movement is within normal volatility range
    - BEARISH (2): Price has fallen significantly over lookback period

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look BACK for regime classification
        forward_window: IGNORED - kept for API compatibility

    Returns:
        Array of regime labels: 0=BULLISH, 1=RANGING, 2=BEARISH
    """
    close = df['close'].values
    n = len(close)
    labels = np.ones(n, dtype=np.int32)  # Default to RANGING (1)

    # Calculate ATR for threshold normalization
    if 'atr' in df.columns:
        atr_vals = df['atr'].values
    else:
        # Simple ATR proxy
        high_low = df['high'].values - df['low'].values if 'high' in df.columns else np.abs(np.diff(close, prepend=close[0]))
        atr_vals = pd.Series(high_low).rolling(14, min_periods=1).mean().values

    # Calculate rolling returns and volatility for regime detection
    returns = pd.Series(close).pct_change()
    rolling_return = returns.rolling(window=lookback, min_periods=lookback).sum().values
    rolling_vol = returns.rolling(window=lookback, min_periods=lookback).std().values

    # Look BACKWARD to determine regime for each bar (no future information)
    for i in range(lookback, n):
        # Skip if we don't have valid volatility
        if np.isnan(rolling_vol[i]) or rolling_vol[i] < 1e-10:
            continue

        # Normalize return by volatility (similar to Sharpe-like metric)
        normalized_return = rolling_return[i] / (rolling_vol[i] * np.sqrt(lookback))

        # Use ATR-based thresholds for consistency with original approach
        # But now based on PAST performance, not FUTURE
        # Symmetric thresholds since we're not trying to balance future outcomes
        if normalized_return > 1.5:  # Strong uptrend over lookback
            labels[i] = 0  # BULLISH
        elif normalized_return < -1.5:  # Strong downtrend over lookback
            labels[i] = 2  # BEARISH
        # else: RANGING (stays as 1)

    # First lookback bars default to RANGING (not enough history)
    labels[:lookback] = 1

    return labels.astype(np.int32)


# =============================================================================
# Volatility Regime Detection (v36 OOD Fix) - DEPRECATED
# =============================================================================
# WARNING: v36 features use ROLLING WINDOWS that adapt to new data.
# After ~500 bars of OOS data, the baseline IS the new regime - no shift detected!
# Use v37 training-anchored features instead (src/data/ood_features.py).
# =============================================================================

def compute_volatility_regime(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 100
) -> pd.Series:
    """
    Classify current volatility regime relative to historical norms.

    This feature gives the agent explicit awareness of which volatility
    environment it's operating in, enabling regime-adaptive behavior.

    Uses z-score bands to classify:
    - 0 = Low volatility (z < -0.5)
    - 1 = Normal volatility (-0.5 <= z <= 0.5)
    - 2 = High volatility (z > 0.5)

    Args:
        df: OHLCV DataFrame with 'close' column
        short_window: Window for current volatility measurement
        long_window: Window for historical baseline

    Returns:
        Series with values 0 (Low), 1 (Normal), 2 (High)
    """
    # Compute rolling volatility (standard deviation of returns)
    vol = df['close'].pct_change().rolling(short_window, min_periods=5).std()

    # Compute baseline volatility statistics over longer window
    vol_ma = vol.rolling(long_window, min_periods=20).mean()
    vol_std = vol.rolling(long_window, min_periods=20).std()

    # Z-score of current volatility relative to historical baseline
    vol_z = (vol - vol_ma) / (vol_std + 1e-8)

    # Classify into regimes using z-score bands
    regime = pd.Series(1, index=df.index, dtype=np.float32)  # Default: Normal
    regime[vol_z < -0.5] = 0  # Low volatility
    regime[vol_z > 0.5] = 2   # High volatility

    return regime.astype(np.float32)


def compute_distribution_shift_score(
    df: pd.DataFrame,
    baseline_window: int = 500,
    current_window: int = 50
) -> pd.Series:
    """
    Compute a score indicating how different current market is from recent baseline.

    High score (close to 1) = current market deviates significantly from baseline,
    indicating potential out-of-distribution conditions where the model may
    be less reliable.

    Combines multiple signals:
    1. Volatility shift (current vs baseline)
    2. Return distribution shift (skewness change)
    3. ATR ratio change

    Args:
        df: DataFrame with 'close' and optionally 'atr' columns
        baseline_window: Lookback for computing baseline statistics
        current_window: Recent window for current statistics

    Returns:
        Series with values 0-1 (0 = similar to baseline, 1 = very different)
    """
    shift_components = []

    # 1. Volatility shift
    vol = df['close'].pct_change().rolling(20, min_periods=5).std()
    vol_baseline = vol.rolling(baseline_window, min_periods=50).mean()
    vol_current = vol.rolling(current_window, min_periods=10).mean()
    vol_shift = np.abs(vol_current - vol_baseline) / (vol_baseline + 1e-8)
    shift_components.append(vol_shift.clip(0, 2) / 2)  # Normalize to 0-1

    # 2. Return distribution shift (skewness change)
    returns = df['close'].pct_change()
    skew_baseline = returns.rolling(baseline_window, min_periods=50).skew()
    skew_current = returns.rolling(current_window, min_periods=10).skew()
    skew_shift = np.abs(skew_current - skew_baseline)
    shift_components.append((skew_shift / 3.0).clip(0, 1))  # Normalize typical skew range

    # 3. Price momentum shift (mean return change)
    mean_ret_baseline = returns.rolling(baseline_window, min_periods=50).mean()
    mean_ret_current = returns.rolling(current_window, min_periods=10).mean()
    ret_baseline_std = returns.rolling(baseline_window, min_periods=50).std()
    momentum_shift = np.abs(mean_ret_current - mean_ret_baseline) / (ret_baseline_std + 1e-8)
    shift_components.append(momentum_shift.clip(0, 3) / 3)  # Normalize to 0-1

    # Combine into single score (average of components)
    shift_score = pd.concat(shift_components, axis=1).mean(axis=1)

    return shift_score.clip(0, 1).astype(np.float32)


# =============================================================================
# Mean Reversion Indicators
# =============================================================================

def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Bollinger Bands %B and Bandwidth.

    %B: Position within bands (0 = lower band, 1 = upper band)
        - %B < 0: Price below lower band (oversold)
        - %B > 1: Price above upper band (overbought)

    Bandwidth: (Upper - Lower) / Middle, measures volatility squeeze

    Args:
        df: OHLCV DataFrame
        period: SMA period (default 20 = standard)
        num_std: Standard deviations for bands (default 2.0)

    Returns:
        Tuple of (percent_b, bandwidth) as float32 Series
    """
    middle = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    # %B: position within bands
    band_width = upper - lower
    percent_b = (df['close'] - lower) / (band_width.replace(0, 1e-10))
    percent_b = percent_b.clip(-1, 2).astype(np.float32)

    # Bandwidth: volatility measure (normalized by middle band)
    bandwidth = (band_width / middle.replace(0, 1e-10)).clip(0, 0.05).astype(np.float32)

    return percent_b, bandwidth


def price_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Z-Score of price relative to rolling mean.

    Measures how many standard deviations price is from its mean.
    - Positive: Price above mean (overbought tendency)
    - Negative: Price below mean (oversold tendency)

    Args:
        df: OHLCV DataFrame
        period: Rolling window size (default 20)

    Returns:
        Z-score Series clipped to [-3, 3], float32
    """
    mean = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std().replace(0, 1e-10)
    zscore = (df['close'] - mean) / std
    return zscore.clip(-3, 3).astype(np.float32)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R momentum oscillator.

    Measures current close relative to high-low range over period.
    Normalized to [-1, 0] for consistency with other features.

    - Near 0: Overbought (close near high)
    - Near -1: Oversold (close near low)

    Args:
        df: OHLCV DataFrame
        period: Lookback period (default 14)

    Returns:
        Williams %R normalized to [-1, 0], float32
    """
    highest = df['high'].rolling(period).max()
    lowest = df['low'].rolling(period).min()
    range_hl = highest - lowest
    wr = (highest - df['close']) / (range_hl.replace(0, 1e-10))
    # Negate to get -1 to 0 range (traditional Williams %R is 0 to -100)
    return (-wr).clip(-1, 0).astype(np.float32)


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Classic momentum oscillator measuring speed of price changes.
    - RSI > 0.7 (70): Overbought
    - RSI < 0.3 (30): Oversold

    Normalized to [0, 1] (from original 0-100).

    Args:
        df: OHLCV DataFrame
        period: RSI period (default 14)

    Returns:
        RSI normalized to [0, 1], float32
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    # Use Wilder's smoothing (exponential moving average)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi_val = 100 - (100 / (1 + rs))
    return (rsi_val / 100).clip(0, 1).astype(np.float32)


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> pd.Series:
    """
    Stochastic %K oscillator (smoothed).

    Measures current close position within recent high-low range.
    - > 0.8: Overbought zone
    - < 0.2: Oversold zone

    Args:
        df: OHLCV DataFrame
        k_period: %K lookback period (default 14)
        d_period: %D smoothing period (default 3)

    Returns:
        Smoothed %K normalized to [0, 1], float32
    """
    lowest = df['low'].rolling(k_period).min()
    highest = df['high'].rolling(k_period).max()
    range_hl = highest - lowest
    k = (df['close'] - lowest) / (range_hl.replace(0, 1e-10))
    # Smooth with D period
    k_smooth = k.rolling(d_period).mean()
    return k_smooth.clip(0, 1).astype(np.float32)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index.

    Measures current price deviation from statistical mean.
    - CCI > 1 (100 original): Strong uptrend / overbought
    - CCI < -1 (-100 original): Strong downtrend / oversold

    Normalized by dividing by 100 and clipping to [-2, 2].

    Args:
        df: OHLCV DataFrame
        period: CCI period (default 20)

    Returns:
        CCI normalized to [-2, 2], float32
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(period).mean()
    # Mean absolute deviation
    mean_dev = typical_price.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci_val = (typical_price - sma_tp) / (0.015 * mean_dev + 1e-10)
    return (cci_val / 100).clip(-2, 2).astype(np.float32)


def rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    pivot_left: int = 5,
    pivot_right: int = 5,
    range_lower: int = 5,
    range_upper: int = 48,
    decay_half_life: int = 12,
) -> pd.Series:
    """
    Detect price vs RSI divergence using pivot-based comparison.

    Matches TradingView's RSI Divergence Indicator logic:
    1. Find pivot highs/lows in RSI (not price) using left/right lookback
    2. Compare current RSI pivot to previous RSI pivot
    3. Compare price at current pivot to price at previous pivot
    4. Divergence = price and RSI moving in opposite directions at pivots

    Regular Bullish:  Price lower low  + RSI higher low  (reversal up signal)
    Regular Bearish:  Price higher high + RSI lower high  (reversal down signal)
    Hidden Bullish:   Price higher low  + RSI lower low   (continuation up)
    Hidden Bearish:   Price lower high  + RSI higher high  (continuation down)

    Output is a decaying continuous signal for RL compatibility:
    +1.0 at bullish divergence, -1.0 at bearish, exponential decay over time.

    Args:
        df: OHLCV DataFrame
        rsi_period: RSI calculation period
        pivot_left: Bars to the left for pivot confirmation
        pivot_right: Bars to the right for pivot confirmation
        range_lower: Minimum bars between pivots
        range_upper: Maximum bars between pivots
        decay_half_life: Half-life of signal decay in bars

    Returns:
        Series: Decaying divergence signal (-1 to +1)
    """
    rsi_vals = (rsi(df, rsi_period) * 100).values  # 0-100 scale
    low_arr = df['low'].values
    high_arr = df['high'].values
    length = len(df)

    decay_rate = 0.5 ** (1.0 / decay_half_life)
    result = np.zeros(length, dtype=np.float32)
    current_signal = 0.0

    # --- Find pivots in RSI ---
    pivot_low_bars = np.full(length, False)
    pivot_high_bars = np.full(length, False)

    total_window = pivot_left + pivot_right + 1
    for i in range(total_window - 1, length):
        center = i - pivot_right  # the candidate pivot bar
        if center < pivot_left:
            continue

        center_rsi = rsi_vals[center]
        if np.isnan(center_rsi):
            continue

        # Check pivot low: center RSI <= all surrounding bars
        window_start = center - pivot_left
        window_end = center + pivot_right + 1
        window = rsi_vals[window_start:window_end]

        if np.any(np.isnan(window)):
            continue

        is_low = True
        is_high = True
        for j in range(len(window)):
            if j == pivot_left:  # skip center
                continue
            if window[j] <= center_rsi:
                is_low = False
            if window[j] >= center_rsi:
                is_high = False
            if not is_low and not is_high:
                break

        if is_low:
            pivot_low_bars[i] = True
        if is_high:
            pivot_high_bars[i] = True

    # --- Track previous pivots and detect divergences ---
    prev_pivot_low_bar = -1
    prev_pivot_low_rsi = np.nan
    prev_pivot_low_price = np.nan

    prev_pivot_high_bar = -1
    prev_pivot_high_rsi = np.nan
    prev_pivot_high_price = np.nan

    for i in range(length):
        # Decay existing signal
        current_signal *= decay_rate

        if pivot_low_bars[i]:
            center = i - pivot_right
            curr_rsi = rsi_vals[center]
            curr_price = low_arr[center]

            if not np.isnan(prev_pivot_low_rsi):
                bars_between = center - prev_pivot_low_bar
                if range_lower <= bars_between <= range_upper:
                    # Regular Bullish: price lower low, RSI higher low
                    if curr_price < prev_pivot_low_price and curr_rsi > prev_pivot_low_rsi:
                        current_signal = 1.0

                    # Hidden Bullish: price higher low, RSI lower low
                    elif curr_price > prev_pivot_low_price and curr_rsi < prev_pivot_low_rsi:
                        current_signal = 0.5  # weaker signal for hidden

            prev_pivot_low_bar = center
            prev_pivot_low_rsi = curr_rsi
            prev_pivot_low_price = curr_price

        if pivot_high_bars[i]:
            center = i - pivot_right
            curr_rsi = rsi_vals[center]
            curr_price = high_arr[center]

            if not np.isnan(prev_pivot_high_rsi):
                bars_between = center - prev_pivot_high_bar
                if range_lower <= bars_between <= range_upper:
                    # Regular Bearish: price higher high, RSI lower high
                    if curr_price > prev_pivot_high_price and curr_rsi < prev_pivot_high_rsi:
                        current_signal = -1.0

                    # Hidden Bearish: price lower high, RSI higher high
                    elif curr_price < prev_pivot_high_price and curr_rsi > prev_pivot_high_rsi:
                        current_signal = -0.5  # weaker signal for hidden

            prev_pivot_high_bar = center
            prev_pivot_high_rsi = curr_rsi
            prev_pivot_high_price = curr_price

        result[i] = current_signal

    return pd.Series(result, index=df.index, dtype=np.float32)


# =============================================================================
# Complete Feature Engineering
# =============================================================================

def engineer_all_features(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    include_sr_features: bool = True,
    training_baseline: Optional['TrainingBaseline'] = None
) -> pd.DataFrame:
    """
    Apply all feature engineering to OHLCV DataFrame.

    IMPORTANT: Does NOT drop NaN rows - alignment is handled by the pipeline.
    This preserves index alignment between timeframes.

    v37 OOD Fix: If training_baseline is provided, computes training-anchored
    OOD features that compare against FIXED training statistics (never adapts).
    Otherwise falls back to v36 rolling-window features (adapts to new data).

    Args:
        df: OHLCV DataFrame
        config: Optional configuration dict
        include_sr_features: Whether to include S/R distance features (slower)
        training_baseline: Optional TrainingBaseline for v37 anchored OOD detection

    Returns:
        DataFrame with all features added (may contain NaN at edges)
    """
    if config is None:
        # v35 FIX: Made feature parameters more regime-robust
        config = {
            'fractal_window': 5,
            'sr_lookback': 100,
            'sma_period': 20,    # v35: Reduced from 50 to 20 for faster adaptation to regime changes
            'ema_fast': 12,
            'ema_slow': 26,
            'chop_period': 14,
            'adx_period': 14,
            'atr_period': 14,
            # Mean Reversion Settings
            'bb_period': 20,
            'bb_std': 2.0,
            'williams_period': 14,
            'rsi_period': 14,
            'divergence_lookback': 10,
        }

    result = df.copy()
    logger.info("Engineering features...")

    # ATR (needed for other features)
    result['atr'] = atr(df, config['atr_period'])
    # Volatility context anchor (preserve absolute scale without Z-score).
    result['atr_context'] = np.log1p(result['atr']).astype(np.float32)

    # v35 FIX: Add percentile-based ATR for regime-robust volatility context
    # This tells the agent where current volatility is relative to recent history
    # (0 = lowest volatility in lookback, 1 = highest volatility in lookback)
    atr_lookback = 288  # ~24 hours of 5m bars
    atr_rolling_rank = result['atr'].rolling(window=atr_lookback, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
    )
    result['atr_percentile'] = atr_rolling_rank.astype(np.float32)

    # Price Action Patterns (pinbar, engulfing, doji removed — no predictive value on 5m)

    # Trend Filters
    result['sma_distance'] = sma_distance(df, result['atr'], config['sma_period'])
    result['ema_gap'] = ema_gap(df, result['atr'], config['ema_fast'], config['ema_slow'])
    result['ema_acceleration'] = ema_acceleration(df, result['atr'], config['ema_fast'], config['ema_slow'])
    result['ema_crossover_recency'] = ema_crossover_recency(df, config['ema_fast'], config['ema_slow'])

    # v35 FIX: Add percentile-based SMA distance for regime-robust trend context
    # In bull markets, price is persistently above SMA causing z-scores to clip at +5
    # Percentile (0-1) tells the agent where current distance is relative to recent history
    sma_lookback = 200  # ~16 hours of 5m bars
    sma_dist_rolling_rank = result['sma_distance'].rolling(window=sma_lookback, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
    )
    result['sma_distance_percentile'] = sma_dist_rolling_rank.astype(np.float32)

    # Regime Detection
    result['chop'] = choppiness_index(df, config['chop_period'])
    result['adx'] = adx(df, config['adx_period'])

    # v35 FIX: Add adaptive choppiness that normalizes based on recent history
    # This makes the "choppy" vs "trending" classification regime-robust
    # Instead of fixed thresholds (38.2/61.8), uses rolling percentile
    chop_lookback = 200
    chop_rolling_rank = result['chop'].rolling(window=chop_lookback, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
    )
    result['chop_percentile'] = chop_rolling_rank.astype(np.float32)

    # S/R Channel Features — TradingView-style with bar-touch strength
    if include_sr_features:
        logger.info("  Computing S/R channel features (adaptive zones, bar-touch strength)...")
        dist_to_r, dist_to_s, str_r, str_s = distance_to_nearest_sr(
            df['close'], df, result['atr'],
            fractal_window=21,   # ~1.75h swings (prd=10 equivalent)
            lookback=290,        # ~24h of history for channel building
            max_channels=6,      # top 6 non-overlapping channels
            channel_width_pct=4.5,  # 4.5% of 300-bar price range (wider for US30 round-number S/R)
        )
        result['dist_to_resistance'] = dist_to_r.clip(0, 50).astype(np.float32)
        result['dist_to_support'] = dist_to_s.clip(0, 50).astype(np.float32)
        result['sr_strength_r'] = str_r.astype(np.float32)
        result['sr_strength_s'] = str_s.astype(np.float32)

    # Market Sessions
    result = add_market_sessions(result)

    # Structure Features (BOS/CHoCH) — v2: larger fractals (n=21) + continuous signals
    struct_fractal_n = 21  # ~1.75h swings on 5m — more meaningful than n=5
    f_high, f_low = detect_fractals(df, n=struct_fractal_n)
    struct_df = detect_structure_breaks(df, f_high, f_low, n=struct_fractal_n)
    for col in struct_df.columns:
        result[col] = struct_df[col]

    # Returns (for normalization and targets)
    result['returns'] = df['close'].pct_change().astype(np.float32)

    # Volatility
    result['volatility'] = result['returns'].rolling(20).std().astype(np.float32)

    # v35 FIX: Add percentile-based volatility for regime-robust vol context
    # Raw volatility scales differently across regimes (2023 vol is higher than 2019-2022)
    # Percentile (0-1) tells the agent where current vol is relative to recent history
    vol_lookback = 200  # ~16 hours of 5m bars
    vol_rolling_rank = result['volatility'].rolling(window=vol_lookback, min_periods=20).apply(
        lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
    )
    result['volatility_percentile'] = vol_rolling_rank.astype(np.float32)

    # ==========================================================================
    # Mean Reversion Indicators
    # ==========================================================================
    result['bb_percent_b'], result['bb_bandwidth'] = bollinger_bands(
        df,
        period=int(config.get('bb_period', 20)),
        num_std=float(config.get('bb_std', 2.0)),
    )
    result['williams_r'] = williams_r(df, int(config.get('williams_period', 14)))
    result['rsi'] = rsi(df, int(config.get('rsi_period', 14)))
    result['rsi_divergence'] = rsi_divergence(
        df,
        rsi_period=int(config.get('rsi_period', 14)),
    )
    logger.info("  Added mean reversion features (BB %B, BB bandwidth, Williams %R, RSI, RSI divergence)")

    # ==========================================================================
    # OOD Features: v37 training-anchored vs v36 rolling-window
    # ==========================================================================
    if training_baseline is not None:
        from src.data.ood_features import compute_training_anchored_ood_features
        result = compute_training_anchored_ood_features(result, training_baseline)
        logger.info("  Added v37 training-anchored OOD features (volatility_vs_training, returns_skew_shift, atr_vs_training, ood_score)")
    else:
        result['volatility_vs_training'] = 0.0
        result['returns_skew_shift'] = 0.0
        result['atr_vs_training'] = 0.0
        result['ood_score'] = 0.0
        logger.info("  Added v37 OOD feature placeholders (zeros - no training baseline provided)")

    # DO NOT drop NaN rows - alignment is handled by the pipeline
    # This preserves index alignment between timeframes
    nan_count = result.isna().any(axis=1).sum()
    logger.info(f"Features complete. {len(result):,} rows, {nan_count:,} rows with NaN (will be aligned in pipeline)")

    # Ensure all float32
    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    return result


def create_smoothed_target(
    df: pd.DataFrame,
    future_window: int = 24,
    smooth_window: int = 24,
    scale_factor: float = 100.0
) -> pd.Series:
    """
    Create the smoothed future return target for Analyst training.

    Target = ((smoothed future close / current close) - 1) * scale_factor

    This teaches the model sustained momentum, not noise.
    
    FIXED: 
    - Added min_periods=1 to reduce NaN data loss at edges.
    - Scale by 100 to get PERCENTAGE returns (prevents mode collapse to near-zero)
    
    Without scaling, targets are ~0.0001 and the model learns trivial solution
    of always predicting 0. With scale_factor=100, targets are ~0.01 (1% moves)
    which forces the model to learn real patterns.

    Args:
        df: DataFrame with 'close' column
        future_window: How many candles ahead
        smooth_window: Rolling window for smoothing
        scale_factor: Multiply returns by this (100 = percentage returns)

    Returns:
        Series of target values (in percentage if scale_factor=100)
    """
    # Use min_periods=1 to reduce NaN values at the edges
    future_smoothed = df['close'].shift(-future_window).rolling(smooth_window, min_periods=1).mean()
    target = (future_smoothed / df['close']) - 1
    
    # Scale to percentage returns to prevent mode collapse
    # Without this, targets are ~0.0001 and model predicts ~0 for everything
    target = target * scale_factor

    return target.astype(np.float32)


def create_return_classes(
    target: pd.Series,
    class_std_thresholds: Tuple = (-0.5, 0.5),
    train_end_idx: Optional[int] = None
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Convert a continuous smoothed-return target into discrete classes.

    Supports both 3-class and 5-class schemes based on threshold count:

    3-class scheme (2 thresholds: down_thresh, up_thresh):
        0: Down      (< down_thresh * std)
        1: Neutral   [down_thresh * std, up_thresh * std]
        2: Up        (> up_thresh * std)

    5-class scheme (4 thresholds: strong_down, weak_down, weak_up, strong_up):
        0: Strong Down   (< strong_down * std)
        1: Weak Down     [strong_down * std, weak_down * std)
        2: Neutral       [weak_down * std, weak_up * std]
        3: Weak Up       (weak_up * std, strong_up * std]
        4: Strong Up     (> strong_up * std)

    Args:
        target: Smoothed return series (already scaled)
        class_std_thresholds: Multipliers of target std that define boundaries
                             2 values for 3-class, 4 values for 5-class

    Returns:
        Tuple of (class labels Series with NaNs preserved, metadata dict)
    """
    # IMPORTANT: To prevent look-ahead bias, compute std on TRAINING portion only
    if train_end_idx is not None:
        std_source = target.iloc[:train_end_idx].dropna()
    else:
        std_source = target.dropna()

    target_std = float(std_source.std())
    num_thresholds = len(class_std_thresholds)

    if num_thresholds == 2:
        # 3-class scheme: Down / Neutral / Up
        down_thresh = class_std_thresholds[0] * target_std
        up_thresh = class_std_thresholds[1] * target_std

        def _assign_class(value: float) -> float:
            if pd.isna(value):
                return np.nan
            if value < down_thresh:
                return 0  # Down
            if value <= up_thresh:
                return 1  # Neutral
            return 2  # Up

        meta = {
            'target_std': target_std,
            'num_classes': 3,
            'down_threshold': down_thresh,
            'up_threshold': up_thresh,
            # Legacy keys for compatibility
            'strong_down_threshold': down_thresh,
            'weak_down_threshold': down_thresh,
            'weak_up_threshold': up_thresh,
            'strong_up_threshold': up_thresh
        }

    elif num_thresholds == 4:
        # 5-class scheme: Strong Down / Weak Down / Neutral / Weak Up / Strong Up
        boundaries = {
            'strong_down': class_std_thresholds[0] * target_std,
            'weak_down': class_std_thresholds[1] * target_std,
            'weak_up': class_std_thresholds[2] * target_std,
            'strong_up': class_std_thresholds[3] * target_std
        }

        def _assign_class(value: float) -> float:
            if pd.isna(value):
                return np.nan
            if value < boundaries['strong_down']:
                return 0
            if value < boundaries['weak_down']:
                return 1
            if value <= boundaries['weak_up']:
                return 2
            if value <= boundaries['strong_up']:
                return 3
            return 4

        meta = {
            'target_std': target_std,
            'num_classes': 5,
            'strong_down_threshold': boundaries['strong_down'],
            'weak_down_threshold': boundaries['weak_down'],
            'weak_up_threshold': boundaries['weak_up'],
            'strong_up_threshold': boundaries['strong_up']
        }
    else:
        raise ValueError(f"class_std_thresholds must have 2 or 4 values, got {num_thresholds}")

    labels = target.apply(_assign_class).astype(np.float32)

    return labels, meta


def create_binary_direction_target(
    df: pd.DataFrame,
    future_window: int = 16,
    smooth_window: int = 12,
    min_move_atr: float = 0.3,
    atr_period: int = 14
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Create binary Up/Down labels, excluding weak/neutral moves.

    This addresses the directional confusion problem where the model achieves
    71% recall on Neutral but only ~50% on Up/Down. By excluding weak moves,
    we train only on clear directional signals.

    Args:
        df: DataFrame with 'close' column (and optionally 'atr')
        future_window: How many candles ahead (16 = 4 hours on 15m)
        smooth_window: Rolling window for smoothing
        min_move_atr: Minimum move in ATR units to count as directional
        atr_period: ATR calculation period if 'atr' not in df

    Returns:
        Tuple of:
            - labels: 0=Down, 1=Up (NaN for excluded neutral moves)
            - valid_mask: Boolean mask for training samples
            - meta: Metadata dict with thresholds and stats
    """
    # Calculate smoothed future return
    future_smoothed = df['close'].shift(-future_window).rolling(
        smooth_window, min_periods=1
    ).mean()
    future_return = (future_smoothed / df['close']) - 1

    # Get or calculate ATR
    if 'atr' in df.columns:
        atr = df['atr']
    else:
        # Calculate ATR from price
        high_low = df['high'] - df['low'] if 'high' in df.columns else df['close'].diff().abs()
        atr = high_low.rolling(atr_period, min_periods=1).mean()

    # Normalize ATR to percentage terms (like future_return)
    atr_pct = atr / df['close']

    # Threshold: at least min_move_atr * ATR move
    threshold = min_move_atr * atr_pct

    # Create labels: 0=Down, 1=Up, NaN=Excluded (neutral/weak)
    labels = pd.Series(index=df.index, dtype=np.float32)
    labels[:] = np.nan  # Start all as NaN (excluded)

    down_mask = future_return < -threshold
    up_mask = future_return > threshold

    labels[down_mask] = 0  # Down
    labels[up_mask] = 1    # Up
    # Neutral moves (between -threshold and +threshold) remain NaN

    # Valid mask for filtering dataset
    valid_mask = ~labels.isna()

    # Metadata
    n_down = down_mask.sum()
    n_up = up_mask.sum()
    n_neutral = len(df) - n_down - n_up
    n_valid = valid_mask.sum()

    meta = {
        'num_classes': 2,
        'min_move_atr': min_move_atr,
        'future_window': future_window,
        'smooth_window': smooth_window,
        'n_down': int(n_down),
        'n_up': int(n_up),
        'n_neutral_excluded': int(n_neutral),
        'n_valid': int(n_valid),
        'pct_excluded': float(n_neutral / len(df) * 100),
        'class_balance': float(n_up / (n_down + n_up)) if (n_down + n_up) > 0 else 0.5
    }

    logger.info(
        f"Binary direction target: {n_down} Down, {n_up} Up, "
        f"{n_neutral} excluded ({meta['pct_excluded']:.1f}%)"
    )

    return labels, valid_mask, meta


def create_auxiliary_targets(
    df: pd.DataFrame,
    future_window: int = 16,
    atr_period: int = 14,
    adx_threshold: float = 25.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Create auxiliary targets for multi-task learning:
    1. Volatility target: Future ATR / Current ATR (regression)
    2. Regime target: Trending (1) vs Ranging (0) based on ADX

    These auxiliary losses provide regularization and help the model
    learn better representations.

    Args:
        df: DataFrame with OHLC data
        future_window: How many candles ahead
        atr_period: ATR calculation period
        adx_threshold: ADX value above which market is "trending"

    Returns:
        Tuple of (volatility_target, regime_target)
    """
    # Calculate ATR
    if 'atr' in df.columns:
        atr = df['atr']
    else:
        high_low = df['high'] - df['low'] if 'high' in df.columns else df['close'].diff().abs()
        atr = high_low.rolling(atr_period, min_periods=1).mean()

    # Volatility target: Future ATR / Current ATR
    future_atr = atr.shift(-future_window)
    volatility_target = (future_atr / atr).fillna(1.0).astype(np.float32)
    # Clip extreme values
    volatility_target = volatility_target.clip(0.5, 2.0)

    # Regime target: 1 if ADX > threshold (trending), else 0
    if 'adx' in df.columns:
        regime_target = (df['adx'] > adx_threshold).astype(np.float32)
    else:
        # Simple proxy: use ATR percentile
        atr_rolling_pct = atr.rolling(100, min_periods=20).apply(
            lambda x: (x[-1] > np.percentile(x, 60)).astype(float),
            raw=True
        )
        regime_target = atr_rolling_pct.fillna(0.0).astype(np.float32)

    return volatility_target, regime_target


# =============================================================================
# Market Sessions
# =============================================================================

def add_market_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market session flags (London, NY, Asian).
    
    Assumes index is DatetimeIndex in UTC.
    
    Sessions (approx UTC):
    - Asian: 00:00 - 09:00
    - London: 08:00 - 17:00
    - NY: 13:00 - 22:00
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        DataFrame with added session columns
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex. Skipping session features.")
        return df
        
    hours = df.index.hour
    
    # Asian Session (Tokyo/Sydney): ~00:00 to 09:00 UTC
    df['session_asian'] = ((hours >= 0) & (hours < 9)).astype(int)
    
    # London Session: ~08:00 to 17:00 UTC
    df['session_london'] = ((hours >= 8) & (hours < 17)).astype(int)
    
    # New York Session: ~13:00 to 22:00 UTC
    df['session_ny'] = ((hours >= 13) & (hours < 22)).astype(int)
    
    return df


def detect_structure_breaks(
    df: pd.DataFrame,
    fractal_highs: pd.Series,
    fractal_lows: pd.Series,
    n: int = 5
) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH) — v2.

    Outputs continuous, RL-friendly features instead of sparse binary pulses:
    - structure_fade: Mean-reversion signal (-1 to +1). Positive after bullish breaks
      (expect fade down), negative after bearish breaks (expect fade up). Decays
      exponentially with half-life of 12 bars (~1h on 5m).
    - bars_since_bos: Normalized time since last BOS (0=just happened, 1=long ago).
      Capped at 200 bars, scaled to [0, 1].
    - bars_since_choch: Same as above for CHoCH events.
    - bos_magnitude: Size of BOS break in ATR units. Positive for bullish, negative
      for bearish. Decays with same half-life as structure_fade.
    - choch_magnitude: Same as above for CHoCH events.
    - bos_streak: Consecutive same-direction BOS count. +3 = 3 bullish BOS in a row
      (strong uptrend), -2 = 2 bearish BOS (downtrend). Resets on CHoCH.
      Normalized to [-1, +1] by dividing by streak_cap (5).

    Args:
        df: OHLCV DataFrame with 'close', 'high', 'low', and 'atr' columns
        fractal_highs: Boolean series of confirmed fractal highs
        fractal_lows: Boolean series of confirmed fractal lows
        n: Window size used for fractal detection (to find price level)

    Returns:
        DataFrame with columns: structure_fade, bars_since_bos, bars_since_choch,
                                bos_magnitude, choch_magnitude, bos_streak
    """
    length = len(df)
    half_n = n // 2
    decay_hl = 12  # half-life in bars (~1h on 5m)
    decay_rate = 0.5 ** (1.0 / decay_hl)
    max_bars = 200  # cap for bars_since normalization
    streak_cap = 5  # normalize streak to [-1, +1] by dividing by this

    # Pre-extract arrays for speed
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    has_atr = 'atr' in df.columns
    atr_arr = df['atr'].values if has_atr else np.ones(length, dtype=np.float64)

    # Output arrays
    structure_fade = np.zeros(length, dtype=np.float32)
    bars_since_bos_arr = np.full(length, max_bars, dtype=np.float32)
    bars_since_choch_arr = np.full(length, max_bars, dtype=np.float32)
    bos_mag_arr = np.zeros(length, dtype=np.float32)
    choch_mag_arr = np.zeros(length, dtype=np.float32)
    bos_streak_arr = np.zeros(length, dtype=np.float32)

    # State variables
    last_high = np.nan
    last_low = np.nan
    last_high_broken = False
    last_low_broken = False
    trend = 0  # 1=Bullish, -1=Bearish, 0=Unknown

    # Running state for continuous features
    current_fade = 0.0
    bars_since_bos = max_bars
    bars_since_choch = max_bars
    current_bos_mag = 0.0
    current_choch_mag = 0.0
    streak_raw = 0  # positive = consecutive bullish BOS, negative = bearish

    # Convert to numpy for faster access
    fh_arr = fractal_highs.values
    fl_arr = fractal_lows.values

    for i in range(length):
        # 1. Update structure points from confirmed fractals
        if fh_arr[i]:
            idx = i - half_n
            if idx >= 0:
                last_high = high_arr[idx]
                last_high_broken = False

        if fl_arr[i]:
            idx = i - half_n
            if idx >= 0:
                last_low = low_arr[idx]
                last_low_broken = False

        # 2. Decay existing signals
        current_fade *= decay_rate
        current_bos_mag *= decay_rate
        current_choch_mag *= decay_rate
        bars_since_bos = min(bars_since_bos + 1, max_bars)
        bars_since_choch = min(bars_since_choch + 1, max_bars)

        # 3. Check for breaks
        close = close_arr[i]
        atr_val = atr_arr[i] if has_atr else 1.0
        if np.isnan(atr_val) or atr_val < 1e-10:
            atr_val = 1.0

        # Break High
        if not np.isnan(last_high) and not last_high_broken and close > last_high:
            last_high_broken = True
            magnitude = (close - last_high) / atr_val

            if trend == 1:
                # BOS bullish (continuation)
                bars_since_bos = 0
                current_bos_mag = magnitude
                current_fade = 1.0  # Bullish break → expect fade down
                streak_raw = max(streak_raw + 1, 1)
            elif trend == -1:
                # CHoCH bullish (reversal)
                bars_since_choch = 0
                current_choch_mag = magnitude
                current_fade = 1.0
                trend = 1
                streak_raw = 1  # reset to +1 on reversal
            else:
                trend = 1
                bars_since_bos = 0
                current_bos_mag = magnitude
                current_fade = 1.0
                streak_raw = 1

        # Break Low
        if not np.isnan(last_low) and not last_low_broken and close < last_low:
            last_low_broken = True
            magnitude = (last_low - close) / atr_val  # positive magnitude

            if trend == -1:
                # BOS bearish (continuation)
                bars_since_bos = 0
                current_bos_mag = -magnitude
                current_fade = -1.0  # Bearish break → expect fade up
                streak_raw = min(streak_raw - 1, -1)
            elif trend == 1:
                # CHoCH bearish (reversal)
                bars_since_choch = 0
                current_choch_mag = -magnitude
                current_fade = -1.0
                trend = -1
                streak_raw = -1  # reset to -1 on reversal
            else:
                trend = -1
                bars_since_bos = 0
                current_bos_mag = -magnitude
                current_fade = -1.0
                streak_raw = -1

        # 4. Write outputs
        structure_fade[i] = current_fade
        bars_since_bos_arr[i] = bars_since_bos / max_bars  # normalize to [0, 1]
        bars_since_choch_arr[i] = bars_since_choch / max_bars
        bos_mag_arr[i] = current_bos_mag
        choch_mag_arr[i] = current_choch_mag
        bos_streak_arr[i] = np.clip(streak_raw / streak_cap, -1.0, 1.0)

    return pd.DataFrame({
        'structure_fade': structure_fade,
        'bars_since_bos': bars_since_bos_arr,
        'bars_since_choch': bars_since_choch_arr,
        'bos_magnitude': bos_mag_arr,
        'choch_magnitude': choch_mag_arr,
        'bos_streak': bos_streak_arr,
    }, index=df.index)


def get_feature_columns(include_ohlcv: bool = False) -> List[str]:
    """Get canonical model feature column names (per timeframe)."""
    from src.live.bridge_constants import MODEL_FEATURE_COLS

    features = list(MODEL_FEATURE_COLS)
    if include_ohlcv:
        features = ['open', 'high', 'low', 'close'] + features
    return features
