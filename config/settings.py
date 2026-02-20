"""
Configuration settings for the Hybrid US30 Trading System.

This module contains all hyperparameters, paths, and constants.
Optimized for Apple M2 Silicon with 8GB RAM constraints.
"""

import os
import torch
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def get_device() -> torch.device:
    """Get the optimal device for computation."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU/MPS memory cache and run garbage collection."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@dataclass
class PathConfig:
    """Path configurations for data and model storage."""

    # Base directory (relative to project root)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Training data directory (configurable via US30_DATA_DIR env var)
    training_data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("US30_DATA_DIR", str(Path.home() / "Desktop" / "Oanda data"))
        )
    )

    @property
    def data_raw(self) -> Path:
        return self.training_data_dir

    @property
    def data_processed(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def models_analyst(self) -> Path:
        return self.base_dir / "models" / "analyst"

    @property
    def models_agent(self) -> Path:
        return self.base_dir / "models" / "agent"

    def ensure_dirs(self):
        """Create all necessary directories."""
        for path in [self.data_raw, self.data_processed,
                     self.models_analyst, self.models_agent]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Data file names
    raw_file: str = "new us30_UTC.csv"
    processed_file: str = "us30_processed.parquet"
    datetime_format: str = "ISO8601"  # Auto-detect or ISO format

    # Timeframes (use lowercase 'h' for pandas 2.0+ compatibility)
    # UPDATED: Changed from 15m/1h/4h to 5m/15m/45m for faster trading analysis
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        '5m': '5min',
        '15m': '15min',
        '45m': '45min'
    })

    # Lookback windows (number of candles)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # Rule: lookback > prediction horizon to avoid temporal confusion
    # Subsample ratios: 15m = 3x base (5m), 45m = 9x base (5m)
    lookback_windows: Dict[str, int] = field(default_factory=lambda: {
        '5m': 48,    # 4 Hours - 2x prediction horizon (proper context)
        '15m': 16,   # 4 Hours - captures trading session
        '45m': 6     # 4.5 Hours - captures trend
    })

    # Train/validation/test splits
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Memory-efficient chunk size for processing
    chunk_size: int = 100_000


@dataclass
class NormalizationConfig:
    """Rolling window normalization configuration."""

    # Rolling window size for Z-score normalization (20 days of 5m bars)
    rolling_window_size: int = 5760

    # Minimum samples before normalization becomes active
    min_rolling_samples: int = 100

    # Clipping value for normalized features (prevents extreme outliers)
    clip_value: float = 5.0

    # Tail size for bridge data buffer
    tail_size: int = 1000

    # Price tolerance for floating point comparisons
    price_tolerance: float = 1e-4


@dataclass
class OODConfig:
    """
    v37 Out-of-Distribution (OOD) detection and adaptation configuration.

    Unlike v36 rolling-window features that adapt to new data, v37 uses
    FIXED training statistics for true OOD detection.
    """
    # Enable/disable OOD-based adaptations
    use_ood_detection: bool = True

    # OOD-based Position Sizing
    # ood_score=0 -> 100% position, ood_score=1 -> min_ratio position
    ood_size_reduction_factor: float = 0.8  # How aggressively to reduce
    min_position_size_ratio: float = 0.2  # Minimum 20% position when fully OOD

    # Warning thresholds for logging
    ood_warning_threshold: float = 0.4  # Warn when mean OOD score > this
    ood_critical_threshold: float = 0.7  # Critical alert when mean OOD score > this


@dataclass
class DataSplitConfig:
    """
    Explicit date-based data split configuration (v37 FIX).

    Provides reproducible splits that don't shift with data alignment.
    These dates take precedence over percentage-based splits when set.

    Default dates assume US30 data spanning 2019-2025:
    - Training: 2019-01-01 to 2022-12-31 (~70%)
    - Validation: 2023-01-01 to 2023-09-30 (~15%)
    - Test (held-out): 2023-10-01 onwards (~15%)
    """

    # Use explicit dates instead of percentage splits
    use_date_splits: bool = True

    # Training data ends here (exclusive)
    train_end_date: str = "2025-10-31"

    # Validation/OOS data ends here (exclusive)
    # Everything after this is TRUE held-out test data
    validation_end_date: str = "2025-12-30"

    # Split ratios (used when use_date_splits=False)
    # These match DataConfig.train_ratio/val_ratio/test_ratio
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10

    # Normalization statistics are computed on training data only
    # This is enforced in both training and backtest pipelines
    normalize_on_train_only: bool = True


@dataclass
class AnalystConfig:
    """Market Analyst configuration (supports both Transformer and TCN)."""
    # Architecture selection - TCN is more stable for binary classification
    architecture: str = "tcn"   # "transformer" or "tcn"

    # Shared architecture settings
    d_model: int = 48           # Hidden dimension (48 > 29 input features, good capacity)
    nhead: int = 6              # Transformer only: attention heads (48/6 = 8 dim/head)
    num_layers: int = 3         # Transformer only: encoder layers
    dim_feedforward: int = 192  # Transformer only: FFN hidden dim (4x d_model)
    dropout: float = 0.4        # v36 OOD FIX: Increased from 0.3 for better regularization
    context_dim: int = 48       # Output context vector dimension (Matched to d_model)

    # Input noise regularization (training only)
    # Adds small Gaussian noise to input features during Analyst training to reduce overfitting.
    # 0.0 disables. Typical useful range: 0.002–0.01 on normalized features.
    input_noise_std: float = 0.05

    # TCN-specific settings
    tcn_num_blocks: int = 4     # Number of residual blocks (dilations: 1, 2, 4, 8)
    tcn_kernel_size: int = 3    # Convolution kernel size

    batch_size: int = 128       # Keep at 128
    learning_rate: float = 3e-4 # Increased for faster learning, no decay
    weight_decay: float = 1e-4  # FIXED: Was 1e-2 (100x too high!) - standard value
    max_epochs: int = 100
    patience: int = 5  # Reduced for faster iteration

    cache_clear_interval: int = 50

    # v9 FIX: TARGET DEFINITION - reduced horizon and smoothing for more predictable signal
    future_window: int = 24      # 2 Hours (24 * 5m) - shorter = more predictable
    smooth_window: int = 12      # 1 Hour (12 * 5m) - less smoothing = preserves signal

    # Binary classification mode
    num_classes: int = 2        # Binary: 0=Down, 1=Up
    use_binary_target: bool = True  # Use binary direction target
    min_move_atr_threshold: float = 1.5  # v9 FIX: Was 0.3 - lower = 4x more training data

    # Auxiliary losses (multi-task learning)
    # RE-ENABLED - easier tasks (regime ~70%, volatility ~65%) provide
    # stronger gradients to shared encoder, reducing overfitting and improving direction
    use_auxiliary_losses: bool = True   # Enabled for multi-task learning
    aux_volatility_weight: float = 0.2  # Volatility prediction (MSE)
    aux_regime_weight: float = 0.4      # INCREASED - Regime is easier, stronger gradients

    # Gradient accumulation for smoother updates (effective batch = batch_size * steps)
    gradient_accumulation_steps: int = 2  # Effective batch size = 128 * 2 = 256

    # Legacy 3-class config (kept for compatibility)
    class_std_thresholds: Tuple[float, float] = (-0.15, 0.15)

    # Input Lookback Windows (Must match DataConfig)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # UPDATED: Changed from 15m/1h/4h to 5m/15m/45m
    lookback_5m: int = 48       # 4 Hours - 6x prediction horizon (proper context)
    lookback_15m: int = 16      # 4 Hours - captures trading session
    lookback_45m: int = 6       # 4.5 Hours - captures trend


@dataclass
class InstrumentConfig:
    """Instrument-specific parameters for US30 (Oanda CFD)."""
    name: str = "US30"
    pip_value: float = 1.0           # 1 point = 1.0 price movement (NOT 0.1 tick size)
    lot_size: float = 1.0            # CFD lot ($1 per point per lot)
    point_multiplier: float = 1.0    # PnL: points × pip_value × lot_size × multiplier = $1/point


@dataclass
class TradingConfig:
    """Trading environment configuration for US30."""
    # Toggle Market Analyst usage
    # If False, agent trains with only raw market features (no analyst context/metrics)
    use_analyst: bool = False  # Agent uses analyst context in observation

    spread_pips: float = 50.0    # Intentional worst-case spread for robust training
    slippage_pips: float = 0.0  # US30 slippage

    # Confidence filtering: Only take trades when agent probability >= threshold
    min_action_confidence: float = 0.0  # Filter low-confidence trades (0.0 = disabled)

    # NEW: Risk-Based Sizing (Not Fixed Lots)
    risk_multipliers: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    
    # NEW: ATR-Based Stops (Not Fixed Pips)
    # SL at 2.0x ATR, TP at 6.0x ATR = 1:3 R/R ratio
    # Break-even at 25% win rate, profitable at ~30%+ win rate
    sl_atr_multiplier: float = 2.0   # SL at 2.0x ATR
    tp_atr_multiplier: float = 6.0   # TP at 6.0x ATR = 1:3 R/R ratio
    
    # Risk Limits
    max_position_size: float = 5.0
    
    # Dollar-Based Position Sizing
    risk_per_trade: float = 100.0  # Dollar risk per trade (if use_percentage=False)
    
    # NEW: Dynamic Risk Sizing
    risk_use_percentage: bool = True  # Use % of Equity instead of fixed $
    risk_percent: float = 5.0         # Risk 1.0% of Equity per trade
    
    # Reward Params (calibrated for US30)
    # US30 has ~100-200 point daily range vs EURUSD ~50-100 pip range
    # reward_scaling = 0.01 means 100 points = 1.0 reward (similar magnitude to EURUSD)
    # v26 FIX: FOMO as OPPORTUNITY COST (not fixed per-step penalty)
    # This penalizes sitting flat during major market moves, encouraging the agent
    # to capture trends. Capped to prevent extreme penalties that cause panic trades.
    opportunity_cost_multiplier: float = 0.2  # Reduced from 5.0 - gentle nudge
    opportunity_cost_cap: float = 0.2         # Reduced from 0.3

    fomo_threshold_atr: float = 4.0  # Trigger on 2.0×ATR moves
    fomo_lookback_bars: int = 24     # Check over 24 bars (2 hours)
    chop_threshold: float = 80.0     # Only extreme chop triggers penalty
    reward_scaling: float = 0.01    # 1.0 reward per 100 points (US30 calibration)

    # Symmetric PnL Scaling
    profit_scaling: float = 0.01     # Symmetric with loss_scaling
    loss_scaling: float = 0.01       # 1.0x for losses (same as reward_scaling)

    # Alpha-Based Reward - rewards OUTPERFORMANCE vs market, not absolute PnL
    # DISABLED alpha baseline - was creating insurmountable barrier with 50 pip spread
    # abs(market_move) * 0.3 + 50 pip spread = impossible to get positive alpha on entry
    # Result: Agent collapsed to 99% flat with 0 trades
    use_alpha_reward: bool = True    # Keep True (controls reward scaling logic)
    alpha_baseline_exposure: float = 0.0  # Disabled: 0.0 = no baseline comparison (was causing flat collapse with 50 pip spread)

    # Trade entry bonus: Disabled. Full spread cost visible to agent for realistic learning.
    trade_entry_bonus: float = 0.0

    # Underwater Decay Penalty - penalizes holding losing trades
    # Encourages cutting losses early, teaching agent to manage drawdown
    underwater_penalty_coef: float = 0.5    # Strong penalty for holding losers
    underwater_threshold_atr: float = 1.5     # Only penalize losses beyond this ATR threshold
    
    # Forced Minimum Hold Time
    # v36 FIX: ENABLED to prevent 1-bar scalping exploitation
    # While action blocking creates PPO gradient mismatch, the alternative (1-bar scalping)
    # is worse. Agent learns that early exit actions are ineffective and adjusts policy.
    # Profit-based early exit override (early_exit_profit_atr) allows exiting big winners.
    min_hold_bars: int = 0  # Minimum bars to hold (0 = disabled)
    early_exit_profit_atr: float = 0.0  # Allow early exit if profit > 3x ATR (overrides min_hold_bars)
    break_even_atr: float = 2.0  # Move SL to break-even when profit reaches 2x ATR
    
    # These are mostly unused now but keep for compatibility if needed
    use_stop_loss: bool = True
    use_take_profit: bool = True
    
    # Environment settings
    # One trading week ≈ 5 days of 5m bars: 5 * 24 * 60 / 5 = 1440 steps.
    max_steps_per_episode: int = 1440
    initial_balance: float = 10000.0
    rolling_norm_min_samples: int = 1  # Use rolling normalization from the first observation
    
    # Validation
    noise_level: float = 0.05  # Moderate regularization noise

    # NEW: "Full Eyes" Agent Features
    agent_lookback_window: int = 12   # Increased to 12 as requested (60 mins of 5m bars)
    include_structure_features: bool = True  # Agent sees BOS/CHoCH

    # v36 OOD FIX: Volatility Augmentation for distribution shift robustness
    # Scales volatility-related features during training to expose agent to
    # different market regimes, improving generalization to new data
    use_volatility_augmentation: bool = True
    volatility_augment_ratio: float = 0.3  # Fraction of samples to augment


@dataclass
class EntropyScheduleConfig:
    """Entropy coefficient schedule for PPO training.

    Gradually reduces exploration over training to stabilize policy.
    4-phase schedule: high exploration → medium → low → minimal.
    """
    phase1_steps: int = 20_000_000      # Steps in phase 1 (high entropy)
    phase1_ent: float = 0.02            # Entropy coefficient for phase 1
    phase2_steps: int = 280_000_000     # Steps in phase 2
    phase2_ent: float = 0.01            # Entropy coefficient for phase 2
    phase3_steps: int = 300_000_000     # Steps in phase 3
    phase3_ent: float = 0.005           # Entropy coefficient for phase 3
    phase4_ent: float = 0.002           # Final entropy coefficient


@dataclass
class BridgeConfig:
    """Runtime configuration for MT5 live trading bridge."""
    host: str = "127.0.0.1"
    port: int = 5555
    socket_max_payload: int = 50_000_000
    main_symbol: str = "US30"
    decision_tf_minutes: int = 5        # US30 uses 5m decision timeframe

    # Persist incoming M1 bars so restarts don't require MT5 bootstrap.
    history_dir: Path = field(default_factory=lambda: Path("data") / "live")
    max_m1_rows: int = 60 * 24 * 30     # ~30 days

    # Minimum M1 rows required before trading is enabled.
    min_m1_rows: int = 60 * 24 * 30
    min_history_days: int = 30

    # Execution mapping (EA expects lots).
    lot_scale: float = 1.0

    # Safety / testing
    dry_run: bool = False

    # Feature pipeline window sizes (recompute on a tail window for speed).
    tail_5m_bars: int = 600
    tail_15m_bars: int = 400
    tail_45m_bars: int = 260

    # Custom model path (optional - overrides default final_model.zip)
    model_path: Optional[Path] = None


@dataclass
class AgentConfig:
    """PPO Sniper Agent configuration."""

    # PPO hyperparameters
    # Previous config had broken minibatch ratio (8192/1024=8 minibatches) which
    # caused gradient noise with sparse rewards. Now using standard PPO setup.
    learning_rate: float = 3e-4  # Increased for faster learning
    n_steps: int = 2048         # Steps per env per rollout
    batch_size: int = 512       # 16384/512 = 32 minibatches (n_envs=8)
    n_epochs: int = 4           # Reduced from 10 to prevent overfitting
    # Standard discount factor for trading (trade-level rewards, not episode-end)
    gamma: float = 0.99  # Reduced for better EV learning
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02       # Reduced exploration for steadier policies
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 1_000_000_000  # 1B steps (Increased back to 1B)

    # Policy network
    # [64, 64] may bottleneck for 49-dim input with 12-action output
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])

    # v36 OOD FIX: Policy network dropout for regularization
    policy_dropout: float = 0.1  # Dropout rate for policy network

    # Vectorized environment settings for faster training
    n_envs: int = 8              # Parallel environments (DummyVecEnv, shared memory)
    use_subproc: bool = False    # DummyVecEnv (shared memory, no per-process duplication)


@dataclass
class FeatureConfig:
    """Feature engineering configuration for US30."""

    # Market structure
    fractal_window: int = 5           # Williams fractal window
    sr_lookback: int = 100            # S/R level lookback

    # Trend indicators
    sma_period: int = 50
    ema_fast: int = 14
    ema_slow: int = 50

    # Regime indicators
    chop_period: int = 14
    adx_period: int = 14
    atr_period: int = 14

    # Volatility sizing reference (US30 calibration)
    risk_pips_target: float = 75.0    # Reference risk ~50 points (was 15 for EURUSD)

    # Mean Reversion Indicators
    bb_period: int = 20              # Bollinger Bands period
    bb_std: float = 2.0              # Bollinger Bands std deviation
    williams_period: int = 14        # Williams %R period (14 = faster signals for US30 momentum)
    rsi_period: int = 14             # RSI period
    divergence_lookback: int = 10    # RSI divergence lookback


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)  # v37: Date-based splits
    ood: OODConfig = field(default_factory=OODConfig)  # v37: OOD detection config
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    entropy_schedule: EntropyScheduleConfig = field(default_factory=EntropyScheduleConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Global settings
    seed: int = 42
    dtype: torch.dtype = torch.float32  # NEVER use float64 on M2
    device: torch.device = field(default_factory=get_device)

    def __post_init__(self):
        """Ensure directories exist and set random seeds."""
        self.paths.ensure_dirs()
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)


# Global configuration instance
config = Config()
