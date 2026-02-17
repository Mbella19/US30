"""
MT5 ↔ Python bridge for live inference.

Design goals:
- Match the offline training/backtest conditions as closely as possible:
  - Ingest 1-minute OHLC from MT5 (server time) + UTC offset
  - Convert timestamps to UTC (timezone-naive like the training pipeline)
  - Rebuild 5m/15m/45m bars using the same pandas resampling semantics
    (label='right', closed='left') to avoid look-ahead bias
  - Apply the same feature engineering and saved normalizers
  - Build observations with the same ordering/scaling as TradingEnv
  - Run frozen Analyst + PPO Agent inference and return trade instructions

The MT5-side EA is expected to connect via TCP, send a length-prefixed JSON
payload, then read a length-prefixed JSON response.
"""

from __future__ import annotations

import json
import socketserver
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config.settings import Config, BridgeConfig, get_device
from src.agents.sniper_agent import SniperAgent
from src.agents.recurrent_agent import RecurrentSniperAgent
from src.data.features import engineer_all_features
from src.data.normalizer import FeatureNormalizer
from src.data.resampler import resample_all_timeframes, align_timeframes
from src.models.analyst import load_analyst
from src.utils.logging_config import setup_logging, get_logger

from .bridge_constants import MODEL_FEATURE_COLS, MARKET_FEATURE_COLS, POSITION_SIZES
from src.data.ood_features import TrainingBaseline

logger = get_logger(__name__)

# Optional visualization support
try:
    from .activation_extractor import ActivationExtractor
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    ActivationExtractor = None


# =============================================================================
# Feature/Observation conventions (must match training)
# =============================================================================


@dataclass(frozen=True)
class MarketFeatureStats:
    """Z-score stats applied to `market_features` inside the observation (FALLBACK only)."""

    cols: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "MarketFeatureStats":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        cols = tuple(data["cols"].tolist())
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        return cls(cols=cols, mean=mean, std=std)


class RollingMarketNormalizer:
    """
    O(1) rolling window normalizer for market features.

    Uses circular buffer + running sums to compute rolling mean/std,
    exactly matching TradingEnv's implementation for training parity.

    Configuration is centralized in config/settings.py NormalizationConfig.
    """

    def __init__(
        self,
        n_features: int,
        fallback_stats: MarketFeatureStats,
        rolling_window_size: Optional[int] = None,
        rolling_min_samples: Optional[int] = None,
        normalization_config=None,
    ):
        self.n_features = n_features
        self.fallback_stats = fallback_stats

        # Import centralized config if not provided
        if normalization_config is None:
            from config.settings import config as default_config
            normalization_config = default_config.normalization

        # Use config values with optional overrides
        self.rolling_window_size = int(
            rolling_window_size if rolling_window_size is not None
            else normalization_config.rolling_window_size
        )
        min_samples = (
            rolling_min_samples if rolling_min_samples is not None
            else normalization_config.min_rolling_samples
        )
        self.rolling_min_samples = max(1, int(min_samples))
        self.clip_value = float(normalization_config.clip_value)

        self.skip_norm_idx = np.array([], dtype=np.int64)
        if "atr_context" in MARKET_FEATURE_COLS:
            base_len = len(MARKET_FEATURE_COLS)
            if base_len > 0 and n_features % base_len == 0:
                ctx_idx = MARKET_FEATURE_COLS.index("atr_context")
                blocks = n_features // base_len
                self.skip_norm_idx = np.array(
                    [ctx_idx + base_len * i for i in range(blocks)],
                    dtype=np.int64
                )
        
        # Circular buffer for O(1) updates
        self.buffer = np.zeros((self.rolling_window_size, n_features), dtype=np.float32)
        self.idx = 0  # Current write position
        self.count = 0  # Samples added (up to rolling_window_size)
        
        # Running sums for O(1) mean/std calculation
        self.rolling_sum = np.zeros(n_features, dtype=np.float64)
        self.rolling_sum_sq = np.zeros(n_features, dtype=np.float64)
    
    def update_and_normalize(self, market_feat_row: np.ndarray) -> np.ndarray:
        """
        Update rolling buffer with new row and return normalized features.
        Uses rolling stats if enough samples, otherwise falls back to global stats.
        """
        market_feat_row = market_feat_row.astype(np.float32)
        
        # Always update the rolling buffer
        if self.count >= self.rolling_window_size:
            # Evict oldest value from running sums
            old_val = self.buffer[self.idx]
            self.rolling_sum -= old_val
            self.rolling_sum_sq -= old_val ** 2
        
        # Add new value
        self.buffer[self.idx] = market_feat_row
        self.rolling_sum += market_feat_row
        self.rolling_sum_sq += market_feat_row ** 2
        
        # Update circular index
        self.idx = (self.idx + 1) % self.rolling_window_size
        if self.count < self.rolling_window_size:
            self.count += 1
        
        # Calculate normalized features
        if self.count >= self.rolling_min_samples:
            # O(1) rolling mean/std calculation
            n = self.count
            rolling_means = (self.rolling_sum / n).astype(np.float32)
            variance = (self.rolling_sum_sq / n) - (rolling_means ** 2)
            rolling_stds = np.maximum(np.sqrt(np.maximum(variance, 0)), 1e-6).astype(np.float32)
            
            normalized = ((market_feat_row - rolling_means) / rolling_stds).astype(np.float32)
        else:
            # Fallback to global stats until we have enough samples
            normalized = ((market_feat_row - self.fallback_stats.mean) / 
                         self.fallback_stats.std).astype(np.float32)
        
        # Safety clip to ±clip_value (matches TradingEnv)
        normalized = np.clip(normalized, -self.clip_value, self.clip_value)
        if self.skip_norm_idx.size:
            normalized[self.skip_norm_idx] = market_feat_row[self.skip_norm_idx]
        return normalized
    
    def warmup(self, historical_rows: np.ndarray) -> None:
        """
        Pre-fill the rolling buffer with historical data.
        Call this at startup with the last N market feature rows.
        """
        if historical_rows is None or len(historical_rows) == 0:
            return
        
        # Take last ROLLING_WINDOW_SIZE rows
        warmup_data = historical_rows[-self.rolling_window_size:].astype(np.float32)
        n_warmup = len(warmup_data)
        
        # Fill buffer
        self.buffer[:n_warmup] = warmup_data
        self.idx = n_warmup % self.rolling_window_size
        self.count = n_warmup
        
        # Compute running sums
        self.rolling_sum = warmup_data.sum(axis=0).astype(np.float64)
        self.rolling_sum_sq = (warmup_data ** 2).sum(axis=0).astype(np.float64)
        
        logger.info(f"Rolling normalizer warmed up with {n_warmup} samples")


def _read_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from a socket."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_length_prefixed_json(sock) -> Dict[str, Any]:
    header = _read_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > 50_000_000:
        raise ValueError(f"Invalid payload length: {length}")
    payload = _read_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def _encode_length_prefixed_json(obj: Dict[str, Any]) -> bytes:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _rates_array_to_df(
    rates: list,
    utc_offset_sec: int,
) -> pd.DataFrame:
    """
    Convert an MT5 rates array to a DataFrame indexed by UTC (timezone-naive).

    Expected per-row formats:
      [time, open, high, low, close] or [time, open, high, low, close, ...]
    Where `time` is in broker/server time; `utc_offset_sec` converts it to UTC.
    """
    if not rates:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    rows = []
    for row in rates:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        t_server = int(row[0])
        t_utc = t_server - int(utc_offset_sec)
        rows.append((t_utc, float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    arr = np.array(rows, dtype=np.float64)
    idx = pd.to_datetime(arr[:, 0].astype(np.int64), unit="s", utc=True).tz_localize(None)
    df = pd.DataFrame(
        {
            "open": arr[:, 1].astype(np.float32),
            "high": arr[:, 2].astype(np.float32),
            "low": arr[:, 3].astype(np.float32),
            "close": arr[:, 4].astype(np.float32),
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _merge_append_ohlc(existing: pd.DataFrame, new_df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing, new_df], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if len(merged) > max_rows:
        merged = merged.iloc[-max_rows:].copy()
    return merged


def _save_history(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Small, fast persistence format
    df.to_parquet(path)


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"History file has invalid index: {path}")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df.sort_index()


def _build_observation(
    *,
    analyst: Optional[torch.nn.Module],  # Now optional
    use_analyst: bool,  # Honor config setting
    agent_env_cfg: Config,
    rolling_normalizer: RollingMarketNormalizer,  # Use rolling normalizer for training parity
    x_5m: np.ndarray,
    x_15m: np.ndarray,
    x_45m: np.ndarray,
    market_feat_row: np.ndarray,
    returns_row_window: np.ndarray,
    position: int,
    entry_price: float,
    current_price: float,
    position_size: float,
    time_in_trade_norm: float = 0.0,  # How long held (0-1)
    price_5_bars_ago: float = 0.0,  # For momentum_aligned hold feature
    current_hour: int = 12,  # For session_progress hold feature (0-23)
    sl_price_override: Optional[float] = None,  # FIX: Override SL for observation after BE triggers
    entry_atr: Optional[float] = None,  # PARITY FIX: Use entry ATR for SL/TP observation
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct an observation vector with the same ordering as TradingEnv._get_observation().
    """
    # FIX: Handle use_analyst=False case (matches TradingEnv logic)
    if use_analyst and analyst is not None:
        device = next(analyst.parameters()).device if hasattr(analyst, "parameters") else torch.device("cpu")

        x_5m_t = torch.tensor(x_5m[None, ...], device=device, dtype=torch.float32)
        x_15m_t = torch.tensor(x_15m[None, ...], device=device, dtype=torch.float32)
        x_45m_t = torch.tensor(x_45m[None, ...], device=device, dtype=torch.float32)

        with torch.no_grad():
            res = analyst.get_probabilities(x_5m_t, x_15m_t, x_45m_t)
            if isinstance(res, (tuple, list)) and len(res) == 3:
                context, probs, _ = res
            else:
                context, probs = res

        context_np = context.cpu().numpy().flatten().astype(np.float32)
        probs_np = probs.cpu().numpy().flatten().astype(np.float32)

        # INFO: Log analyst output statistics (critical for diagnosing flat trading)
        logger.info(
            "ANALYST_OUTPUT | context: min=%.3f max=%.3f mean=%.3f | probs: [%.3f, %.3f]",
            float(context_np.min()), float(context_np.max()), float(context_np.mean()),
            float(probs_np[0]), float(probs_np[1]),
        )

        # Analyst metrics (binary vs multi-class)
        if len(probs_np) == 2:
            p_down = float(probs_np[0])
            p_up = float(probs_np[1])
            confidence = max(p_down, p_up)
            edge = p_up - p_down
            uncertainty = 1.0 - confidence
            analyst_metrics = np.array([p_down, p_up, edge, confidence, uncertainty], dtype=np.float32)
        else:
            p_down = float(probs_np[0])
            p_neutral = float(probs_np[1])
            p_up = float(probs_np[2])
            confidence = float(np.max(probs_np))
            edge = p_up - p_down
            uncertainty = 1.0 - confidence
            analyst_metrics = np.array(
                [p_down, p_neutral, p_up, edge, confidence, uncertainty], dtype=np.float32
            )
    else:
        # No analyst: empty context and metrics (matches TradingEnv when use_analyst=False)
        context_np = np.array([], dtype=np.float32)
        analyst_metrics = np.array([], dtype=np.float32)
        probs_np = np.array([0.5, 0.5], dtype=np.float32)  # Neutral for info dict

    # Position state (mirrors TradingEnv normalization)
    atr = float(market_feat_row[0]) if len(market_feat_row) > 0 else 1.0
    atr_safe = max(atr, 1e-6)

    if position != 0:
        # FIX: Match TradingEnv normalization (divide by atr only, not atr*100)
        if position == 1:
            entry_price_norm = (current_price - entry_price) / atr_safe
        else:
            entry_price_norm = (entry_price - current_price) / atr_safe
        entry_price_norm = float(np.clip(entry_price_norm, -10.0, 10.0))

        pip_value = agent_env_cfg.instrument.pip_value
        if position == 1:
            unrealized_pnl = (current_price - entry_price) / pip_value
        else:
            unrealized_pnl = (entry_price - current_price) / pip_value
        unrealized_pnl *= float(position_size)
        # FIX: Match TradingEnv - normalize by atr_safe, not fixed 100
        unrealized_pnl_norm = float(unrealized_pnl / atr_safe)
    else:
        entry_price_norm = 0.0
        unrealized_pnl_norm = 0.0

    # Position state now has 4 elements (matching TradingEnv)
    position_state = np.array([float(position), entry_price_norm, unrealized_pnl_norm, time_in_trade_norm], dtype=np.float32)

    # Use rolling window normalization for training parity
    # No need to check column ordering - rolling_normalizer handles this
    market_feat_norm = rolling_normalizer.update_and_normalize(market_feat_row)

    # INFO: Log normalized market features (critical for diagnosing flat trading)
    logger.info(
        "MARKET_NORM | 5m: atr=%.2f chop=%.2f adx=%.2f regime=%.2f | "
        "bos_bull=%.2f bos_bear=%.2f | obs_range=[%.2f, %.2f]",
        market_feat_norm[0], market_feat_norm[1], market_feat_norm[2], market_feat_norm[3],
        market_feat_norm[10] if len(market_feat_norm) > 10 else 0,
        market_feat_norm[11] if len(market_feat_norm) > 11 else 0,
        float(market_feat_norm.min()), float(market_feat_norm.max())
    )

    # SL/TP distance features (mirrors TradingEnv)
    # PARITY FIX: Use entry ATR for SL/TP computation (matches enforcement which uses frozen entry ATR)
    sl_tp_atr = float(entry_atr) if entry_atr is not None and entry_atr > 0 else atr
    dist_sl_norm = 0.0
    dist_tp_norm = 0.0
    if position != 0 and sl_tp_atr > 1e-8:
        pip_value = agent_env_cfg.instrument.pip_value
        sl_pips = max((sl_tp_atr * agent_env_cfg.trading.sl_atr_multiplier) / pip_value, 5.0)
        tp_pips = max((sl_tp_atr * agent_env_cfg.trading.tp_atr_multiplier) / pip_value, 5.0)

        if position == 1:
            # FIX: Use sl_price_override after BE triggers (parity with training)
            sl_price = sl_price_override if sl_price_override is not None else (entry_price - sl_pips * pip_value)
            tp_price = entry_price + tp_pips * pip_value
            dist_sl_norm = (current_price - sl_price) / sl_tp_atr
            dist_tp_norm = (tp_price - current_price) / sl_tp_atr
        else:
            # FIX: Use sl_price_override after BE triggers (parity with training)
            sl_price = sl_price_override if sl_price_override is not None else (entry_price + sl_pips * pip_value)
            tp_price = entry_price - tp_pips * pip_value
            dist_sl_norm = (sl_price - current_price) / sl_tp_atr
            dist_tp_norm = (current_price - tp_price) / sl_tp_atr

    # Hold-Encouraging Features (matching TradingEnv)
    profit_progress = 0.0  # 0 = at entry, 1 = at TP
    dist_to_tp_pct = 1.0   # 1 = at entry, 0 = at TP
    momentum_aligned = 0.0  # Positive = price moving in trade direction
    session_progress = current_hour / 24.0  # 0-1 based on hour of day

    if position != 0 and atr > 1e-8:
        tp_atr_mult = float(agent_env_cfg.trading.tp_atr_multiplier)
        tp_target = tp_atr_mult * atr
        pip_value_calc = float(agent_env_cfg.instrument.pip_value)

        # Profit Progress: How far toward TP (0 = entry, 1 = at TP)
        # unrealized_pnl is already calculated above in pip units
        unrealized_price_units = unrealized_pnl * pip_value_calc  # Convert to price units
        profit_progress = float(np.clip(unrealized_price_units / tp_target, -1.0, 1.0))

        # Distance to TP as percentage (1 = at entry, 0 = at TP)
        dist_to_tp_pct = float(np.clip(1.0 - profit_progress, 0.0, 2.0))

        # Momentum Aligned: Is recent price movement in trade direction?
        if price_5_bars_ago > 0:
            price_change = (current_price - price_5_bars_ago) / atr
            momentum_aligned = float(np.clip(price_change * position, -2.0, 2.0))

    hold_features = np.array([
        profit_progress,
        dist_to_tp_pct,
        momentum_aligned,
        session_progress
    ], dtype=np.float32)

    obs = np.concatenate(
        [
            context_np,
            position_state,
            market_feat_norm,
            analyst_metrics,
            np.array([dist_sl_norm, dist_tp_norm], dtype=np.float32),
            hold_features,
        ],
        axis=0,
    ).astype(np.float32)

    # Full-eyes returns window
    # FIX: TradingEnv does NOT scale returns (removed the * 100 that broke PPO)
    lookback = int(agent_env_cfg.trading.agent_lookback_window)
    if lookback > 0:
        if returns_row_window.shape[0] != lookback:
            raise ValueError(f"returns window mismatch: {returns_row_window.shape[0]} != {lookback}")
        obs = np.concatenate([obs, returns_row_window.astype(np.float32)], axis=0)

    info = {
        "p_down": float(probs_np[0]) if probs_np.size >= 1 else 0.5,
        "p_up": float(probs_np[-1]) if probs_np.size >= 2 else 0.5,
    }
    return obs.astype(np.float32), info


class MT5BridgeState:
    def __init__(self, cfg: BridgeConfig, system_cfg: Config, viz_queue=None):
        self.cfg = cfg
        self.system_cfg = system_cfg
        self.viz_queue = viz_queue

        self.history_dir = cfg.history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.m1_path = self.history_dir / f"{cfg.main_symbol}_M1.parquet"
        self.m1: pd.DataFrame = _load_history(self.m1_path)

        # Pre-flight check: Verify all parity-critical artifacts exist BEFORE loading
        # This ensures we fail fast with clear error messages rather than cryptic load failures
        required_artifacts = [
            (system_cfg.paths.models_analyst / "normalizer_5m.pkl", "5m normalizer"),
            (system_cfg.paths.models_analyst / "normalizer_15m.pkl", "15m normalizer"),
            (system_cfg.paths.models_analyst / "normalizer_45m.pkl", "45m normalizer"),
        ]
        missing = [(str(p), name) for p, name in required_artifacts if not p.exists()]
        if missing:
            msg = "PARITY ERROR: Missing required artifacts for training/live parity:\n" + \
                  "\n".join(f"  - {name}: {path}" for path, name in missing)
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Load artifacts
        self.normalizers: Dict[str, FeatureNormalizer] = {
            "5m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_5m.pkl"),
            "15m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_15m.pkl"),
            "45m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_45m.pkl"),
        }

        market_stats_path = system_cfg.paths.models_agent / "market_feat_stats.npz"
        if market_stats_path.exists():
            self.market_feat_stats = MarketFeatureStats.load(market_stats_path)
        else:
            # Create dummy fallback stats (only used before rolling normalizer warms up)
            n_market_feats = len(MARKET_FEATURE_COLS) * 3  # 5m + 15m + 45m
            self.market_feat_stats = MarketFeatureStats(
                cols=tuple(MARKET_FEATURE_COLS * 3),
                mean=np.zeros(n_market_feats, dtype=np.float32),
                std=np.ones(n_market_feats, dtype=np.float32),
            )
            logger.warning("market_feat_stats.npz not found - using zeros/ones fallback (minimal impact with rolling_min_samples=1)")

        # v37: Load training baseline for anchored OOD features
        # PARITY FIX: training_baseline is now REQUIRED for v37 OOD feature parity
        baseline_path = system_cfg.paths.models_agent / "training_baseline.json"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"PARITY ERROR: training_baseline.json not found at {baseline_path}. "
                "This is required for v37 OOD feature parity with training. "
                "Re-run the training pipeline to generate this file."
            )
        try:
            self.training_baseline = TrainingBaseline.load(baseline_path)
            logger.info("v37 TrainingBaseline loaded from %s", baseline_path)
        except Exception as e:
            raise RuntimeError(
                f"PARITY ERROR: Failed to load training_baseline.json: {e}. "
                "This file is corrupted or incompatible. Re-run training pipeline."
            ) from e
        
        # Rolling window normalizer for market features (matches TradingEnv)
        # FIX: Training uses market features across 3 timeframes, not just 1 TF
        rolling_min_samples = getattr(system_cfg.trading, "rolling_norm_min_samples", None)
        self.rolling_normalizer = RollingMarketNormalizer(
            n_features=len(MARKET_FEATURE_COLS) * 3,  # 5m + 15m + 45m
            fallback_stats=self.market_feat_stats,
            rolling_min_samples=rolling_min_samples,
        )
        # FIX: Track if warmup has been done (will be done on first decide() with data)
        self._rolling_warmup_done = False

        feature_dims = {k: len(MODEL_FEATURE_COLS) for k in ("5m", "15m", "45m")}

        # FIX: Honor use_analyst config (training may disable analyst)
        self.use_analyst = getattr(system_cfg.trading, 'use_analyst', True)
        if self.use_analyst:
            analyst_path = system_cfg.paths.models_analyst / "best.pt"
            self.analyst = load_analyst(str(analyst_path), feature_dims, device=system_cfg.device, freeze=True)
            self.analyst.eval()
            # v40 FIX: Verify analyst is actually frozen for production safety
            frozen_params = all(not p.requires_grad for p in self.analyst.parameters())
            if not frozen_params:
                raise RuntimeError(
                    "PARITY ERROR: Analyst model is not frozen! "
                    "This would cause gradient flow during inference, corrupting frozen context assumption."
                )
            logger.info("Analyst ENABLED - loading from %s (verified frozen)", analyst_path)
        else:
            self.analyst = None
            logger.info("Analyst DISABLED (use_analyst=False) - agent using market features only")

        # v40 FIX: Verify all normalizers have consistent column lists
        if len(self.normalizers) == 3:
            cols_5m = set(self.normalizers["5m"].feature_cols)
            cols_15m = set(self.normalizers["15m"].feature_cols)
            cols_45m = set(self.normalizers["45m"].feature_cols)
            if cols_5m != cols_15m or cols_5m != cols_45m:
                mismatch_info = f"5m: {len(cols_5m)}, 15m: {len(cols_15m)}, 45m: {len(cols_45m)}"
                logger.warning(f"Normalizer column mismatch detected: {mismatch_info}")

        obs_dim = self._expected_obs_dim()
        dummy_env = _make_dummy_env(obs_dim)

        # Detect if this is a recurrent model by checking for marker file
        # Check both standard and recurrent directories
        recurrent_marker = system_cfg.paths.models_agent_recurrent / ".recurrent"
        recurrent_model_path = system_cfg.paths.models_agent_recurrent / "final_model.zip"
        standard_model_path = system_cfg.paths.models_agent / "final_model.zip"

        self.is_recurrent = False
        if cfg.model_path is not None:
            # Load custom model checkpoint (always standard PPO, not recurrent)
            custom_path = Path(cfg.model_path)
            if not custom_path.exists():
                raise FileNotFoundError(f"Custom model not found: {custom_path}")
            self.agent = SniperAgent.load(str(custom_path), dummy_env, device="cpu")
            logger.info("Loaded custom PPO agent from %s", custom_path)
        elif recurrent_marker.exists() and recurrent_model_path.exists():
            # Load recurrent model
            self.is_recurrent = True
            self.agent = RecurrentSniperAgent.load(str(recurrent_model_path), dummy_env, device="cpu")
            self.agent.reset_lstm_states()
            logger.info("Loaded RecurrentPPO agent from %s", recurrent_model_path)
        else:
            # Load standard PPO model
            self.agent = SniperAgent.load(str(standard_model_path), dummy_env, device="cpu")
            logger.info("Loaded standard PPO agent from %s", standard_model_path)

        # v40 FIX: Verify loaded model's observation space matches expected dimension
        model_obs_dim = self.agent.model.observation_space.shape[0]
        if model_obs_dim != obs_dim:
            raise ValueError(
                f"PARITY ERROR: Model observation space ({model_obs_dim}) does not match "
                f"expected dimension ({obs_dim}). Model was trained with different configuration. "
                f"Check use_analyst, agent_lookback_window, and feature settings."
            )

        # Track position changes for LSTM episode boundary detection
        self._last_position: int = 0

        self.last_decision_label_utc: Optional[pd.Timestamp] = None
        # Parity: track entry bar for min-hold enforcement (fallback when MT5 omits open_time).
        self._fallback_entry_pos: Optional[int] = None
        self._fallback_entry_label_utc: Optional[pd.Timestamp] = None
        self._fallback_position: int = 0
        # FIX: Track close price at entry for observation parity (training uses close, not fill price)
        self._entry_close_price: Optional[float] = None
        # FIX: Track BE state for observation parity - after BE, sl_price should be entry_close_price
        self._be_triggered: bool = False
        self._sl_price_for_obs: Optional[float] = None

        # Visualization support
        self.activation_extractor = None
        if self.viz_queue is not None and VISUALIZATION_AVAILABLE:
            self.activation_extractor = ActivationExtractor(self.analyst, self.agent)
            logger.info("Visualization enabled - activation extractor initialized")

        logger.info(
            "MT5 bridge ready | symbol=%s | obs_dim=%d | feature_dim=%d",
            cfg.main_symbol,
            obs_dim,
            len(MODEL_FEATURE_COLS),
        )

    def _expected_obs_dim(self) -> int:
        # FIX: Handle use_analyst=False case (matches TradingEnv logic)
        if self.use_analyst and self.analyst is not None:
            context_dim = int(getattr(self.analyst, "context_dim", self.system_cfg.analyst.context_dim))
            analyst_metrics_dim = 5 if int(getattr(self.analyst, "num_classes", 2)) == 2 else 6
        else:
            context_dim = 0
            analyst_metrics_dim = 0

        # FIX: Training uses market features across 3 timeframes, not just 1 TF
        n_market = len(MARKET_FEATURE_COLS) * 3  # 5m + 15m + 45m
        returns_dim = int(self.system_cfg.trading.agent_lookback_window)
        # Position now has 4 elements: [position, entry_price_norm, unrealized_pnl_norm, time_in_trade_norm]
        # FIX: Add 4 for hold features
        return context_dim + 4 + n_market + analyst_metrics_dim + 2 + 4 + returns_dim

    def _snap_utc_offset(self, raw_offset: int) -> int:
        """
        Snap UTC offset to the nearest 30 minutes (1800s).
        This prevents bar index shifting if the user's PC clock drifts slightly
        vs the Broker Server time (since offset = TimeCurrent - TimeGMT).
        """
        if raw_offset == 0:
            return 0
        snap_interval = 1800
        snapped = int(round(raw_offset / snap_interval) * snap_interval)
        if snapped != raw_offset:
            # Log only if significantly different (> 1 minute) to avoid spam
            if abs(snapped - raw_offset) > 60:
                logger.warning(
                    "UTC offset drift detected! Raw: %ds, Snapped: %ds. Check PC clock sync.",
                    raw_offset, snapped
                )
        return snapped

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        raw_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))
        utc_offset_sec = self._snap_utc_offset(raw_offset_sec)

        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or rates.get("M1") or []
        m1_df = _rates_array_to_df(m1_rates, utc_offset_sec=utc_offset_sec)
        if not m1_df.empty:
            self.m1 = _merge_append_ohlc(self.m1, m1_df, self.cfg.max_m1_rows)
            _save_history(self.m1, self.m1_path)

    def _should_decide_now(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[pd.Timestamp]]:
        raw_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))
        utc_offset_sec = self._snap_utc_offset(raw_offset_sec)
        
        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or []
        if not m1_rates:
            return False, None

        last_row = m1_rates[-1]
        if not isinstance(last_row, (list, tuple)) or len(last_row) < 1:
            return False, None

        t_server = int(last_row[0])
        t_utc_open = t_server - utc_offset_sec
        t_utc_close = t_utc_open + 60

        label = pd.to_datetime(t_utc_close, unit="s", utc=True).tz_localize(None)
        
        # Speculative Inference Check:
        # We want to run inference ALWAYS (to drive visualization), but only Trade 
        # when the bar is actually closed relative to our decision interval.
        is_trade_time = False
        if t_utc_close % (self.cfg.decision_tf_minutes * 60) == 0:
            if self.last_decision_label_utc is None or label > self.last_decision_label_utc:
                is_trade_time = True
        
        return is_trade_time, label

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.update_from_payload(payload)

        raw_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))
        utc_offset_sec = self._snap_utc_offset(raw_offset_sec)

        # 1. Update M1 History
        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or []
        m1_df = _rates_array_to_df(m1_rates, utc_offset_sec=utc_offset_sec)
        if not m1_df.empty:
            self.m1 = _merge_append_ohlc(self.m1, m1_df, self.cfg.max_m1_rows)
            # Optimization: Don't save to disk on every single tick, only if significant new data
            if len(m1_df) > 100:
                _save_history(self.m1, self.m1_path)
        t_hist = time.time()

        # Always run inference for visualization (speculative), 
        # but only execute trade if should_decide is True.
        t_start = time.time()
        should_decide, label = self._should_decide_now(payload)
        
        # For visualization, we use the *latest available* label/timestamp
        # even if it's not a closed bar yet.
        if label is None:
             # Fallback to current time snapped to 1m
             rates = payload.get("rates", {})
             m1_rates = rates.get("m1") or rates.get("1m") or []
             if m1_rates:
                 t_server = int(m1_rates[-1][0])
                 # utc_offset_sec is already calculated above
                 t_utc = t_server - utc_offset_sec
                 label = pd.to_datetime(t_utc, unit="s", utc=True).tz_localize(None)
             else:
                 return {"action": 999, "reason": "no_data"}

        # Check sufficiency
        if len(self.m1) < self.cfg.min_m1_rows:
            if self.m1.empty:
                 return {"action": 999, "reason": f"warming_up_m1 ({len(self.m1)}/{self.cfg.min_m1_rows})"}
            span = self.m1.index.max() - self.m1.index.min()
            if span < pd.Timedelta(days=self.cfg.min_history_days):
                 return {"action": 999, "reason": f"warming_up_span ({span})"}

        # 2. Resample Dataframes (Full Rebuild)
        # TODO: Optimize to only rebuild tail if history is large
        df_1m = self.m1.copy()
        resampled = resample_all_timeframes(df_1m, self.system_cfg.data.timeframes)
        df_5m = resampled["5m"]
        df_15m = resampled["15m"]
        df_45m = resampled["45m"]
        t_df = time.time()

        # 3. Feature Engineering
        feature_cfg = {
            "fractal_window": self.system_cfg.features.fractal_window,
            "sr_lookback": self.system_cfg.features.sr_lookback,
            "sma_period": self.system_cfg.features.sma_period,
            "ema_fast": self.system_cfg.features.ema_fast,
            "ema_slow": self.system_cfg.features.ema_slow,
            "chop_period": self.system_cfg.features.chop_period,
            "adx_period": self.system_cfg.features.adx_period,
            "atr_period": self.system_cfg.features.atr_period,
            # Mean Reversion Indicators
            "bb_period": self.system_cfg.features.bb_period,
            "bb_std": self.system_cfg.features.bb_std,
            "williams_period": self.system_cfg.features.williams_period,
            "rsi_period": self.system_cfg.features.rsi_period,
            "divergence_lookback": self.system_cfg.features.divergence_lookback,
        }

        # Optimization: Process only tail for speed.
        # For rolling-norm parity, do a one-time larger tail so the warmup window is full.
        base_tail_size = 1000
        tail_size = base_tail_size
        if not self._rolling_warmup_done:
            warmup_tail_size = self.rolling_normalizer.rolling_window_size + 200
            tail_size = max(base_tail_size, warmup_tail_size)

        if len(df_5m) > tail_size:
            df_5m = df_5m.iloc[-tail_size:].copy()
        if len(df_15m) > tail_size:
            df_15m = df_15m.iloc[-tail_size:].copy()
        if len(df_45m) > tail_size:
            df_45m = df_45m.iloc[-tail_size:].copy()

        # v37: Pass training baseline for anchored OOD features
        df_5m = engineer_all_features(df_5m, feature_cfg, training_baseline=self.training_baseline)
        df_15m = engineer_all_features(df_15m, feature_cfg, training_baseline=self.training_baseline)
        df_45m = engineer_all_features(df_45m, feature_cfg, training_baseline=self.training_baseline)

        # Align higher TFs to 5m grid
        df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

        # Drop invalid
        common_valid = (~df_5m.isna().any(axis=1)) & (~df_15m.isna().any(axis=1)) & (~df_45m.isna().any(axis=1))
        df_5m = df_5m[common_valid]
        df_15m = df_15m[common_valid]
        df_45m = df_45m[common_valid]
        t_feat = time.time()
        
        # Ensure all MARKET_FEATURE_COLS exist before normalizer transform
        # so that df_*_n DataFrames also contain them (FeatureNormalizer preserves non-feature cols)
        for col in MARKET_FEATURE_COLS:
            if col not in df_5m.columns:
                df_5m[col] = 0.0
            if col not in df_15m.columns:
                df_15m[col] = 0.0
            if col not in df_45m.columns:
                df_45m[col] = 0.0

        # Apply NORMALIZERS
        # Apply saved normalizers (per timeframe). RAW columns remain unscaled.
        df_5m_n = self.normalizers["5m"].transform(df_5m)
        df_15m_n = self.normalizers["15m"].transform(df_15m)
        df_45m_n = self.normalizers["45m"].transform(df_45m)

        if label not in df_5m_n.index:
            # Not enough data to form this label in our UTC-resampled timeline.
            return {"action": 999, "reason": "label_not_ready"}

        # Find positional index
        pos = int(df_5m_n.index.get_loc(label))
        
        # Use snapped offset for display/logging correctness
        raw_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))
        utc_offset_sec = self._snap_utc_offset(raw_offset_sec)


        lookback_5m = int(self.system_cfg.analyst.lookback_5m)
        lookback_15m = int(self.system_cfg.analyst.lookback_15m)
        lookback_45m = int(self.system_cfg.analyst.lookback_45m)
        subsample_15m = 3
        subsample_45m = 9
        start_idx = max(
            lookback_5m,
            (lookback_15m - 1) * subsample_15m + 1,
            (lookback_45m - 1) * subsample_45m + 1,
        )

        if pos < start_idx:
            # If we don't have enough history for a full decision, we skip.
            # But for visualization we might want to just zero-pad? 
            # For safety, let's just skip to avoid crashing model.
            return {"action": 999, "reason": "insufficient_lookback"}

        features_5m = df_5m_n[MODEL_FEATURE_COLS].values.astype(np.float32)
        features_15m = df_15m_n[MODEL_FEATURE_COLS].values.astype(np.float32)
        features_45m = df_45m_n[MODEL_FEATURE_COLS].values.astype(np.float32)

        x_5m = features_5m[pos - lookback_5m + 1:pos + 1]

        idx_range_15m = list(
            range(pos - (lookback_15m - 1) * subsample_15m, pos + 1, subsample_15m)
        )
        x_15m = features_15m[idx_range_15m]

        idx_range_45m = list(
            range(pos - (lookback_45m - 1) * subsample_45m, pos + 1, subsample_45m)
        )
        x_45m = features_45m[idx_range_45m]

        # DEBUG: Log analyst feature statistics for diagnosis
        logger.debug(
            "ANALYST_FEATURES | 5m: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f | "
            "15m: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f | "
            "45m: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
            x_5m.shape, x_5m.min(), x_5m.max(), x_5m.mean(), x_5m.std(),
            x_15m.shape, x_15m.min(), x_15m.max(), x_15m.mean(), x_15m.std(),
            x_45m.shape, x_45m.min(), x_45m.max(), x_45m.mean(), x_45m.std(),
        )
        # DEBUG: Log last bar features (most recent) with column names
        last_5m = df_5m_n[MODEL_FEATURE_COLS].iloc[pos]
        logger.debug(
            "LAST_BAR_FEATURES | returns=%.4f vol=%.4f sma_dist=%.4f | "
            "session: asian=%.0f london=%.0f ny=%.0f | "
            "struct_fade=%.2f bars_bos=%.2f bars_choch=%.2f bos_mag=%.2f choch_mag=%.2f",
            float(last_5m.get("returns", 0)), float(last_5m.get("volatility", 0)),
            float(last_5m.get("sma_distance", 0)),
            float(last_5m.get("session_asian", 0)), float(last_5m.get("session_london", 0)),
            float(last_5m.get("session_ny", 0)), float(last_5m.get("structure_fade", 0)),
            float(last_5m.get("bars_since_bos", 0)), float(last_5m.get("bars_since_choch", 0)),
            float(last_5m.get("bos_magnitude", 0)),
            float(last_5m.get("choch_magnitude", 0)),
        )

        # PARITY FIX: Market features from FeatureNorm-transformed DataFrames.
        # Training pipeline: run_pipeline step_3b applies FeatureNorm BEFORE train_agent,
        # so prepare_env_data extracts market features from already-normalized DataFrames.
        # Bridge must do the same: FeatureNorm first, then rolling norm on top.
        market_feat_5m = df_5m_n[MARKET_FEATURE_COLS].iloc[pos].values.astype(np.float32)
        market_feat_15m = df_15m_n[MARKET_FEATURE_COLS].iloc[pos].values.astype(np.float32)
        market_feat_45m = df_45m_n[MARKET_FEATURE_COLS].iloc[pos].values.astype(np.float32)
        market_feat_row = np.concatenate([market_feat_5m, market_feat_15m, market_feat_45m])

        # INFO: Log key market features (always visible for diagnosis)
        logger.info(
            "MARKET_FEAT | 5m: atr=%.1f chop=%.1f adx=%.1f | "
            "struct_fade=%.2f bars_bos=%.2f bos_mag=%.2f choch_mag=%.2f",
            market_feat_5m[0], market_feat_5m[1], market_feat_5m[2],
            market_feat_5m[11] if len(market_feat_5m) > 11 else 0,  # structure_fade
            market_feat_5m[12] if len(market_feat_5m) > 12 else 0,  # bars_since_bos
            market_feat_5m[14] if len(market_feat_5m) > 14 else 0,  # bos_magnitude
            market_feat_5m[15] if len(market_feat_5m) > 15 else 0,  # choch_magnitude
        )

        # FIX: Warmup rolling normalizer on first decide() with historical data
        if not self._rolling_warmup_done and pos >= self.rolling_normalizer.rolling_min_samples:
            # Extract historical market features from all 3 TFs (up to 5760 bars or available)
            warmup_size = min(pos, self.rolling_normalizer.rolling_window_size)
            warmup_start = max(0, pos - warmup_size)
            # PARITY FIX: Use FeatureNorm-transformed DataFrames for warmup (matches training)
            hist_5m = df_5m_n[MARKET_FEATURE_COLS].iloc[warmup_start:pos].values.astype(np.float32)
            hist_15m = df_15m_n[MARKET_FEATURE_COLS].iloc[warmup_start:pos].values.astype(np.float32)
            hist_45m = df_45m_n[MARKET_FEATURE_COLS].iloc[warmup_start:pos].values.astype(np.float32)
            warmup_data = np.concatenate([hist_5m, hist_15m, hist_45m], axis=1)
            self.rolling_normalizer.warmup(warmup_data)
            self._rolling_warmup_done = True
            logger.info(f"Rolling normalizer warmed up with {len(warmup_data)} historical bars")

        # Returns window (already normalized by FeatureNormalizer; env multiplies by 100)
        lookback_ret = int(self.system_cfg.trading.agent_lookback_window)
        returns_series = df_5m_n["returns"].values.astype(np.float32)
        returns_window = returns_series[pos - lookback_ret + 1:pos + 1] if lookback_ret > 0 else np.array([], dtype=np.float32)
        # v40 FIX: Pad with zeros if insufficient data (matches training behavior)
        # Training pads with zeros for early episode steps, bridge should do the same for parity
        if lookback_ret > 0 and returns_window.shape[0] < lookback_ret:
            pad_size = lookback_ret - returns_window.shape[0]
            returns_window = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                returns_window
            ])

        # Position state from MT5
        pos_payload = payload.get("position", {}) or {}
        pos_type = int(pos_payload.get("type", -1))
        mt5_volume = float(pos_payload.get("volume", 0.0))
        mt5_entry = float(pos_payload.get("price", 0.0))
        # Current protective levels (0.0 if not set).
        try:
            mt5_sl = float(pos_payload.get("sl", 0.0) or 0.0)
        except (TypeError, ValueError):
            mt5_sl = 0.0
        try:
            mt5_tp = float(pos_payload.get("tp", 0.0) or 0.0)
        except (TypeError, ValueError):
            mt5_tp = 0.0
        # Optional: MT5 position open time (server epoch seconds). If present, enables exact
        # bar-based min_hold_bars enforcement even across bridge restarts.
        mt5_open_time_server: Optional[int] = None
        if "open_time" in pos_payload:
            try:
                mt5_open_time_server = int(pos_payload.get("open_time") or 0)
            except (TypeError, ValueError):
                mt5_open_time_server = None

        if pos_type == 0:
            position = 1
        elif pos_type == 1:
            position = -1
        else:
            position = 0

        current_price = float(df_5m_n["close"].iloc[pos])
        entry_price = float(mt5_entry) if position != 0 else current_price
        # Keep observation scaling consistent with training env: invert lot_scale so the agent
        # sees the same "position_size" units it was trained on.
        lot_scale = float(self.cfg.lot_scale) if float(self.cfg.lot_scale) > 0 else 1.0
        position_size = float(mt5_volume / lot_scale) if position != 0 else 0.0

        # Parity: compute bars held since entry (5m bars), using MT5 open_time when available.
        entry_pos: Optional[int] = None
        entry_label_utc: Optional[pd.Timestamp] = None
        if position == 0:
            self._fallback_entry_pos = None
            self._fallback_entry_label_utc = None
            self._fallback_position = 0
            self._entry_close_price = None
            # FIX: Reset BE state when position closes
            self._be_triggered = False
            self._sl_price_for_obs = None
        else:
            if mt5_open_time_server is not None and mt5_open_time_server > 0:
                try:
                    open_time_utc = pd.to_datetime(
                        int(mt5_open_time_server) - int(utc_offset_sec), unit="s", utc=True
                    ).tz_localize(None)
                    # Map to the most recent 5m label at-or-before the open time.
                    entry_label = df_5m_n.index.asof(open_time_utc)
                    if entry_label is not None and not pd.isna(entry_label):
                        entry_label_utc = pd.to_datetime(entry_label)
                        entry_pos = int(df_5m_n.index.get_loc(entry_label_utc))
                except Exception:
                    entry_pos = None
                    entry_label_utc = None

            # Fallback: derive entry from first observation of an open position.
            if entry_pos is None:
                if self._fallback_position != position or self._fallback_entry_pos is None:
                    self._fallback_entry_pos = pos
                    self._fallback_entry_label_utc = label
                    # FIX: Track close price at entry for observation parity (training uses close, not MT5 fill)
                    self._entry_close_price = current_price
                self._fallback_position = position
                entry_pos = self._fallback_entry_pos
                entry_label_utc = self._fallback_entry_label_utc
            else:
                # MT5 open_time was available - set entry_close_price from that bar
                if self._entry_close_price is None and entry_pos is not None:
                    try:
                        self._entry_close_price = float(df_5m_n["close"].iloc[int(entry_pos)])
                    except Exception:
                        self._entry_close_price = current_price

        # Fixed-at-entry ATR for SL/TP/BE parity with TradingEnv.
        entry_atr: Optional[float] = None
        if position != 0 and entry_pos is not None:
            try:
                entry_atr = float(df_5m_n["atr"].iloc[int(entry_pos)])
            except Exception:
                entry_atr = None

        # Calculate time in trade (normalized to [0, 1])
        if position != 0 and entry_pos is not None:
            bars_held_for_obs = max(0, int(pos - entry_pos))
            time_in_trade_norm = min(bars_held_for_obs / 100.0, 1.0)
        else:
            time_in_trade_norm = 0.0

        # Get data for hold features
        price_5_bars_ago = float(df_5m["close"].iloc[max(0, pos - 5)]) if pos >= 5 else 0.0
        current_hour = label.hour if hasattr(label, 'hour') else 12

        t_inf_start = time.time()
        # FIX: Use entry_close_price for observation parity (training uses close, not MT5 fill)
        entry_close_price = self._entry_close_price if self._entry_close_price is not None else entry_price

        obs, obs_info = _build_observation(
            analyst=self.analyst,
            use_analyst=self.use_analyst,  # Honor config setting
            agent_env_cfg=self.system_cfg,
            rolling_normalizer=self.rolling_normalizer,  # Use rolling normalizer
            x_5m=x_5m,
            x_15m=x_15m,
            x_45m=x_45m,
            market_feat_row=market_feat_row,
            returns_row_window=returns_window,
            position=position,
            entry_price=entry_close_price,  # FIX: Use close at entry, not MT5 fill
            current_price=current_price,
            position_size=position_size,
            time_in_trade_norm=time_in_trade_norm,  # Pass time in trade
            price_5_bars_ago=price_5_bars_ago,  # For momentum_aligned
            current_hour=current_hour,  # For session_progress
            sl_price_override=self._sl_price_for_obs,  # FIX: Use entry_close_price after BE triggers
            entry_atr=entry_atr,  # PARITY FIX: Use entry ATR for SL/TP observation
        )
        t_obs = time.time()

        if self.cfg.dry_run:
            self.last_decision_label_utc = label
            return {"action": 999, "reason": "dry_run", **obs_info}

        # Detect episode boundary for LSTM state management
        # Episode boundary = position changed from open to flat
        episode_start = (self._last_position != 0 and position == 0)
        self._last_position = position

        if self.is_recurrent:
            # RecurrentPPO needs episode_start flag for LSTM state management
            action, _ = self.agent.predict(
                obs,
                deterministic=True,
                episode_start=episode_start,
                min_action_confidence=float(self.system_cfg.trading.min_action_confidence),
            )
        else:
            # Standard PPO
            action, _ = self.agent.predict(
                obs,
                deterministic=True,
                min_action_confidence=float(self.system_cfg.trading.min_action_confidence),
            )
        t_agent = time.time()

        # LOG TIMING BREAKDOWN
        total_time = t_agent - t_start
        timing_msg = (
            f"Timing Breakdown: Total={total_time:.3f}s | "
            f"Hist={t_hist-t_start:.3f}s | DF={t_df-t_hist:.3f}s | Feat={t_feat-t_df:.3f}s | "
            f"Analyst+Obs={t_obs-t_inf_start:.3f}s | Agent={t_agent-t_obs:.3f}s"
        )
        if total_time > 2.0:
            logger.warning(timing_msg)
        else:
            logger.debug(timing_msg)

        action = np.array(action).astype(np.int32).flatten()
        if action.size < 2:
            return {"action": 999, "reason": "invalid_agent_action"}

        # INFO: Log PPO action probabilities to understand why it chooses Flat
        try:
            obs_tensor, _ = self.agent.model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self.agent.model.policy.get_distribution(obs_tensor)
                # Get direction probabilities (first 3 logits: Flat, Long, Short)
                if hasattr(dist, 'distribution') and hasattr(dist.distribution, '__iter__'):
                    dir_dist = list(dist.distribution)[0]
                    dir_probs = torch.softmax(dir_dist.logits, dim=-1).cpu().numpy().flatten()
                    logger.info(
                        "PPO_PROBS | Flat=%.1f%% Long=%.1f%% Short=%.1f%% | chosen=%d",
                        dir_probs[0] * 100, dir_probs[1] * 100, dir_probs[2] * 100,
                        int(action[0])
                    )
        except Exception as e:
            logger.debug(f"Could not log PPO probs: {e}")

        # DEBUG: Log observation summary for understanding PPO decisions
        logger.debug(
            "PPO_INPUT | obs_shape=%s | obs_min=%.3f max=%.3f mean=%.3f | "
            "context_sum=%.3f market_sum=%.3f",
            obs.shape, float(obs.min()), float(obs.max()), float(obs.mean()),
            float(obs[:32].sum()) if len(obs) >= 32 else 0.0,  # Context vector
            float(obs[36:78].sum()) if len(obs) >= 78 else 0.0,  # Market features (approx)
        )

        direction = int(action[0])
        # v40 FIX: Bounds check for direction (0=Flat, 1=Long, 2=Short)
        if direction not in (0, 1, 2):
            logger.warning(f"Invalid direction {direction} from agent, defaulting to 0 (Flat)")
            direction = 0
        size_idx = int(action[1])
        size_idx = int(np.clip(size_idx, 0, 3))

        # Parity: forced minimum hold time (parity with TradingEnv/backtest: blocks EXIT and FLIPS).
        # Parity: profit-based early exit override
        exit_blocked = False
        bars_held = 0
        min_hold_bars = int(getattr(self.system_cfg.trading, "min_hold_bars", 0))
        early_exit_profit_atr = float(getattr(self.system_cfg.trading, "early_exit_profit_atr", 3.0))
        
        if position != 0 and min_hold_bars > 0 and entry_pos is not None:
            bars_held = max(0, int(pos - entry_pos))
            if bars_held < min_hold_bars:
                would_close_or_flip = (
                    direction == 0 or  # Flat/Exit
                    (position == 1 and direction == 2) or  # Long→Short flip
                    (position == -1 and direction == 1)    # Short→Long flip
                )
                if would_close_or_flip:
                    # Check for profit-based early exit override
                    allow_early_exit = False
                    if early_exit_profit_atr > 0 and entry_price > 0:
                        atr_check = float(market_feat_row[0])
                        pip_value_check = float(self.system_cfg.instrument.pip_value)
                        # FIX: current_price already defined at line 645, no reassignment needed

                        if position == 1:  # Long
                            unrealized_pnl = (current_price - entry_price) / pip_value_check
                        else:  # Short
                            unrealized_pnl = (entry_price - current_price) / pip_value_check
                        
                        profit_threshold = early_exit_profit_atr * atr_check
                        if unrealized_pnl > profit_threshold:
                            allow_early_exit = True
                            obs_info["early_exit_profit"] = True
                            obs_info["profit_pips"] = float(unrealized_pnl)
                    
                    if not allow_early_exit:
                        direction = 1 if position == 1 else 2
                        exit_blocked = True
                        obs_info["exit_blocked"] = True
                        obs_info["bars_held"] = int(bars_held)
                        obs_info["min_hold_bars"] = int(min_hold_bars)
                        if entry_label_utc is not None:
                            obs_info["entry_label_utc"] = int(entry_label_utc.timestamp())

        atr = float(market_feat_row[0])
        pip_value = float(self.system_cfg.instrument.pip_value)

        base_size = float(POSITION_SIZES[size_idx])
        lots = base_size
        
        if bool(getattr(self.system_cfg.trading, "volatility_sizing", True)):
            sl_pips = (atr * float(self.system_cfg.trading.sl_atr_multiplier)) / pip_value
            sl_pips = max(sl_pips, 5.0)

            # Dynamic Risk Sizing (Broker & Account Aware)
            # Extract dynamic inputs from payload
            account_info = payload.get("account", {})
            equity = float(account_info.get("equity", 0.0))
            
            symbol_info = payload.get("symbol", {})
            tick_value = float(symbol_info.get("tick_value", 0.0))
            tick_size = float(symbol_info.get("tick_size", 0.0))

            # Determine Risk Basis (Fixed $ or % Equity)
            use_pct = getattr(self.system_cfg.trading, "risk_use_percentage", False)
            if use_pct and equity > 0:
                 risk_percent = float(getattr(self.system_cfg.trading, "risk_percent", 1.0))
                 risk_basis_amount = equity * (risk_percent / 100.0)
            else:
                 risk_basis_amount = float(getattr(self.system_cfg.trading, "risk_per_trade", 100.0))

            # Apply Agent's Conviction Multiplier (0.5x, 1.0x, 2.0x, etc)
            total_risk_amount = risk_basis_amount * base_size

            # Calculate Lots
            # Formula: Lots = RiskMoney / (SL_Distance_Price * ValuePerPriceUnit)
            # ValuePerPriceUnit = TickValue / TickSize
            sl_dist_price = sl_pips * pip_value
            
            # FIX: Always use training-aligned formula for observation parity
            # The broker's tick_value/tick_size can be ~100x different from training's dollars_per_pip=1.0
            # This affects position_size which feeds into unrealized_pnl_norm observation
            dollars_per_pip = pip_value * float(self.system_cfg.instrument.lot_size) * float(self.system_cfg.instrument.point_multiplier)
            lots = total_risk_amount / max(dollars_per_pip * sl_pips, 1e-6)

            logger.info(
                "TRAINING_ALIGNED_SIZE | Risk=$%.2f SL_pips=%.2f $/pip=%.2f -> Lots=%.4f",
                total_risk_amount, sl_pips, dollars_per_pip, lots
            )

            lots = float(np.clip(lots, 0.01, 100.0))

            # FIX: OOD-based position sizing reduction (mirrors TradingEnv lines 1311-1333)
            # Without this, live gets full exposure while training learns with OOD-reduced sizes
            use_ood_detection = getattr(self.system_cfg.ood, "use_ood_detection", True)
            if use_ood_detection:
                ood_size_reduction_factor = float(getattr(self.system_cfg.ood, "ood_size_reduction_factor", 0.8))
                min_position_size_ratio = float(getattr(self.system_cfg.ood, "min_position_size_ratio", 0.2))

                # Get ood_score from market features (5m block)
                try:
                    if 'ood_score' in MARKET_FEATURE_COLS:
                        ood_idx = MARKET_FEATURE_COLS.index('ood_score')
                        if ood_idx < len(market_feat_row):
                            ood_score = float(market_feat_row[ood_idx])
                            ood_score = float(np.clip(ood_score, 0.0, 1.0))

                            # ood_score=0 -> multiplier=1.0, ood_score=1 -> multiplier=min_ratio
                            ood_multiplier = max(min_position_size_ratio, 1.0 - ood_size_reduction_factor * ood_score)
                            lots *= ood_multiplier

                            logger.info(
                                "OOD_SIZING | score=%.3f multiplier=%.2f adjusted_lots=%.4f",
                                ood_score, ood_multiplier, lots
                            )
                except Exception as e:
                    logger.warning("OOD_SIZING | Failed to apply: %s", e)

        # Apply lot_scale for live trading
        lots = float(np.clip(lots * float(self.cfg.lot_scale), 0.0, 1000.0))

        sl_price = 0.0
        tp_price = 0.0
        break_even_update = False
        current_dir = 1 if position == 1 else 2 if position == -1 else 0

        is_opening_trade = (position == 0 and direction in (1, 2))
        is_flipping_trade = (position != 0 and direction in (1, 2) and direction != current_dir)

        if is_opening_trade or is_flipping_trade:
            # New position: set initial SL/TP based on current ATR (mirrors training/backtest intent).
            sl_pips = max((atr * float(self.system_cfg.trading.sl_atr_multiplier)) / pip_value, 5.0)
            tp_pips = max((atr * float(self.system_cfg.trading.tp_atr_multiplier)) / pip_value, 5.0)
            if direction == 1:
                sl_price = current_price - sl_pips * pip_value
                tp_price = current_price + tp_pips * pip_value
            else:
                sl_price = current_price + sl_pips * pip_value
                tp_price = current_price - tp_pips * pip_value
        elif position != 0 and direction in (1, 2) and direction == current_dir:
            # Existing position: do NOT recompute SL/TP each bar.
            # Only apply break-even SL move when conditions are met (parity with TradingEnv).
            break_even_atr = float(getattr(self.system_cfg.trading, "break_even_atr", 0.0))
            if break_even_atr > 0 and entry_atr is not None and pip_value > 0:
                break_even_profit_pips = (entry_atr * break_even_atr) / pip_value
                if position == 1:
                    unrealized_pips = (current_price - entry_price) / pip_value
                else:
                    unrealized_pips = (entry_price - current_price) / pip_value

                if unrealized_pips >= break_even_profit_pips:
                    price_tol = 1e-4
                    sl_would_tighten = (
                        mt5_sl <= 0.0
                        or (position == 1 and mt5_sl < entry_price - price_tol)
                        or (position == -1 and mt5_sl > entry_price + price_tol)
                    )
                    if sl_would_tighten:
                        sl_price = entry_price
                        tp_price = mt5_tp if mt5_tp > 0.0 else 0.0
                        break_even_update = True
                        obs_info["break_even_update"] = True
                        obs_info["break_even_threshold_pips"] = float(break_even_profit_pips)
                        obs_info["break_even_unrealized_pips"] = float(unrealized_pips)
                        # FIX: Track BE state for observation parity - use entry_close_price for next obs
                        self._be_triggered = True
                        self._sl_price_for_obs = self._entry_close_price if self._entry_close_price is not None else entry_price

        # NOTE: last_decision_label_utc is now set at the START of decide() to prevent race conditions

        logger.info(
            "Decision @ %s | action=%d size=%.2f sl=%.2f tp=%.2f | p_up=%.3f p_down=%.3f%s%s",
            label,
            direction,
            round(lots, 2),
            float(sl_price),
            float(tp_price),
            float(obs_info.get("p_up", 0.5)),
            float(obs_info.get("p_down", 0.5)),
            " | exit_blocked" if exit_blocked else "",
            " | break_even" if break_even_update else "",
        )

        # Queue visualization data if enabled
        if self.viz_queue is not None:
            try:
                action_names = ['FLAT', 'LONG', 'SHORT']
                size_names = ['0.5x', '1.0x', '1.5x', '2.0x']

                # Get action probabilities
                action_probs = {'flat': 0.33, 'long': 0.33, 'short': 0.34}
                try:
                    obs_tensor, _ = self.agent.model.policy.obs_to_tensor(obs)
                    with torch.no_grad():
                        dist = self.agent.model.policy.get_distribution(obs_tensor)
                        if hasattr(dist, 'distribution') and hasattr(dist.distribution, '__iter__'):
                            dir_dist = list(dist.distribution)[0]
                            dir_probs = torch.softmax(dir_dist.logits, dim=-1).cpu().numpy().flatten()
                            action_probs = {
                                'flat': float(dir_probs[0]),
                                'long': float(dir_probs[1]),
                                'short': float(dir_probs[2])
                            }
                        value = self.agent.model.policy.predict_values(obs_tensor)
                        value_estimate = float(value.cpu().numpy().flatten()[0])
                except Exception:
                    value_estimate = 0.0

                # Get agent activations if extractor is available
                agent_activations = {'actor_layers': {}, 'critic_layers': {}}
                if self.activation_extractor is not None:
                    try:
                        agent_activations = self.activation_extractor.extract_agent_activations(obs)
                    except Exception:
                        pass

                viz_data = {
                    'timestamp': time.time(),
                    'market': {
                        'price': float(current_price),
                        'atr': float(atr),
                        'spread': float(self.system_cfg.trading.spread_pips),
                    },
                    'position': {
                        'type': position,
                        'type_name': 'LONG' if position == 1 else 'SHORT' if position == -1 else 'FLAT',
                        'volume': float(mt5_volume),
                        'unrealized_pnl': float(obs_info.get('unrealized_pnl', 0.0)),
                        'entry_price': float(entry_price),
                    },
                    'analyst': {
                        'probabilities': {
                            'p_up': float(obs_info.get('p_up', 0.5)),
                            'p_down': float(obs_info.get('p_down', 0.5)),
                        },
                        'confidence': float(obs_info.get('confidence', 0.5)),
                        'edge': float(obs_info.get('edge', 0.0)),
                    },
                    'agent': {
                        'actor_layers': agent_activations.get('actor_layers', {}),
                        'critic_layers': agent_activations.get('critic_layers', {}),
                        'action_probs': action_probs,
                        'value_estimate': value_estimate,
                    },
                    'decision': {
                        'action': direction,
                        'action_name': action_names[direction] if direction < 3 else 'UNKNOWN',
                        'size_idx': size_idx,
                        'size_name': size_names[size_idx] if size_idx < 4 else 'UNKNOWN',
                        'confidence': float(action_probs.get(action_names[direction].lower(), 0.0)) if direction < 3 else 0.0,
                        'reason': 'ENTRY' if position == 0 else 'HOLD',
                    }
                }
                self.viz_queue.put(viz_data)
            except Exception as e:
                logger.debug(f"Failed to queue visualization data: {e}")

        return {
            "action": direction,
            "size": round(lots, 2),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "ts_utc": int(label.timestamp()),
            **obs_info,
        }


def _make_dummy_env(obs_dim: int):
    import gymnasium as gym
    from gymnasium import spaces

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([3, 4])

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            return np.zeros((obs_dim,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((obs_dim,), dtype=np.float32), 0.0, True, False, {}

    return _DummyEnv()


class _BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        import time
        t0 = time.time()
        try:
            payload = _decode_length_prefixed_json(self.request)
            response = self.server.bridge_state.decide(payload)
            dt = time.time() - t0
            if dt > 2.0:
                logger.warning(f"Bridge decision took {dt:.3f}s (SLOW)")
            else:
                logger.debug(f"Bridge decision took {dt:.3f}s")
        except Exception as e:
            logger.exception("Bridge error during handling: %s", e)
            response = {"action": 999, "reason": "server_error"}

        try:
            encoded_resp = _encode_length_prefixed_json(response)
            self.request.sendall(encoded_resp)
        except Exception as e:
            logger.error("Failed to send response back to MT5: %s", e)


class MT5BridgeServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, bridge_state: MT5BridgeState):
        super().__init__(server_address, handler_class)
        self.bridge_state = bridge_state


def run_mt5_bridge(
    bridge_cfg: BridgeConfig,
    system_cfg: Optional[Config] = None,
    log_dir: Optional[str | Path] = None,
    viz_queue = None,
) -> None:
    if system_cfg is None:
        system_cfg = Config()

    if system_cfg.device is None:
        system_cfg.device = get_device()

    setup_logging(str(log_dir) if log_dir is not None else None, name=__name__)

    state = MT5BridgeState(bridge_cfg, system_cfg, viz_queue=viz_queue)
    server = MT5BridgeServer((bridge_cfg.host, bridge_cfg.port), _BridgeHandler, state)

    logger.info("Listening on %s:%d", bridge_cfg.host, bridge_cfg.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
