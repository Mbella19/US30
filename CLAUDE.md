# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (data → analyst → agent → backtest)
python scripts/core/run_pipeline.py

# Partial runs
python scripts/core/run_pipeline.py --skip-analyst    # Use existing analyst model
python scripts/core/run_pipeline.py --analyst-only    # Train analyst only
python scripts/core/run_pipeline.py --backtest-only   # Backtest only

# Individual training
python -m src.training.train_analyst
python -m src.training.train_agent

# Live trading bridge
python scripts/core/run_mt5_bridge.py --host 127.0.0.1 --port 5555 --symbol US30
```

## Testing

```bash
pytest tests/                           # All tests
pytest tests/test_parity.py -v          # Training ↔ live bridge parity (most critical)
pytest tests/test_trading_env.py -v     # Single test file
pytest tests/ -m "not slow"             # Skip slow tests
```

Key test files: `test_parity.py` (training/bridge observation match), `test_normalization.py` (FeatureNormalizer), `test_alpha_reward.py` (reward shaping), `test_reward_hacking.py` (exploitation detection).

## Architecture

Two-stage hybrid trading system for US30 (Dow Jones CFD):

1. **Market Analyst (Supervised)**: TCN model trained on future returns → outputs frozen 32-dim context vectors
2. **Sniper Agent (RL)**: PPO agent consumes market features only (`use_analyst=False`) → discrete trading decisions
3. **Execution**: TCP bridge to MetaTrader 5 via `ClaudeConnector.mq5` EA

**Multi-Timeframe**: 5m (base/decision), 15m (medium), 45m (trend)

**Pipeline Flow** (`scripts/core/run_pipeline.py`):
```
1-min CSV → Resample 5m/15m/45m → Feature Engineering (2-pass for OOD baseline)
  → FeatureNorm Z-Score (step_3b) → Train Analyst (frozen) → Train PPO Agent → Backtest
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/core/run_pipeline.py` | End-to-end entry point (7 steps) |
| `scripts/core/run_mt5_bridge.py` | Live trading bridge server |
| `scripts/core/prepare_mt5_bridge_artifacts.py` | Creates `market_feat_stats.npz` for bridge fallback |
| `config/settings.py` | All hyperparameters and paths (dataclass-based) |
| `src/environments/trading_env.py` | Gymnasium RL environment |
| `src/agents/sniper_agent.py` | PPO agent (SB3 wrapper) |
| `src/models/tcn_analyst.py` | TCN analyst architecture (default) |
| `src/models/analyst.py` | Transformer analyst (experimental) |
| `src/live/mt5_bridge.py` | TCP server + observation construction for live trading |
| `src/live/bridge_constants.py` | MODEL_FEATURE_COLS (30) and MARKET_FEATURE_COLS (33) |
| `src/data/features.py` | Feature engineering (patterns, indicators, structure, mean reversion) |
| `src/data/normalizer.py` | FeatureNormalizer (Z-score) class |
| `src/evaluation/metrics.py` | Primary metrics (backtest/pipeline) |

## Normalization Pipeline (CRITICAL for parity)

Two-layer normalization that MUST match between training and live bridge:

**Layer 1 — FeatureNormalizer (Z-score)**: Fits on training data only, applied per timeframe. Separate normalizer for 5m/15m/45m (different scales). Clips to ±5.0. Saved as `normalizer_*.pkl`.

**Layer 2 — RollingMarketNormalizer**: Applied in `TradingEnv` during training and in `mt5_bridge` during live inference. O(1) circular buffer implementation. Falls back to pre-computed stats from `market_feat_stats.npz`.

**RAW_COLUMNS** (skipped by FeatureNormalizer, defined in `run_pipeline.py`):
`open, high, low, close, atr, chop, adx, session_asian, session_london, session_ny, atr_context`

**Parity rule**: Both training and bridge must extract MARKET_FEATURE_COLS from **FeatureNorm-transformed** DataFrames (`df_5m_n`, not `df_5m`), then apply rolling normalization on top. The training pipeline applies FeatureNorm in step_3b BEFORE passing DataFrames to `train_agent` → `prepare_env_data`. The bridge must do the same.

## Observation Vector Structure

Constructed in `TradingEnv._get_observation()` and `mt5_bridge._build_observation()`.

**With `use_analyst=False` (121 dims, current config):**

| Component | Dims | Index Range |
|-----------|------|-------------|
| Position | 4 | 0-3 |
| Market 5m | 33 | 4-36 |
| Market 15m | 33 | 37-69 |
| Market 45m | 33 | 70-102 |
| SL/TP | 2 | 103-104 |
| Hold Features | 4 | 105-108 |
| Returns | 12 | 109-120 |

- **position_state**: [position, entry_price_norm, unrealized_pnl_norm, time_in_trade] — normalized by ATR
- **market_feat_norm**: 33 features × 3 timeframes = 99 dims (20 base + 4 percentile + 4 OOD + 5 mean reversion), rolling-normalized
- **sl_tp**: [dist_sl_norm, dist_tp_norm] — normalized by ATR
- **hold_features**: [profit_progress, dist_to_tp_pct, momentum_aligned, session_progress]
- **returns_window**: Last 12 bars of 5m log-returns (from FeatureNorm'd data)

With `use_analyst=True`: prepend context(32), insert analyst_metrics(5) after market → 155 dims.

## Instrument-Specific Configuration

- **Symbol**: US30 (Dow Jones CFD)
- **pip_value**: 1.0 (1 point = 1.0 price movement)
- **lot_size**: 1.0 ($1 per point per lot)
- **spread_pips**: 10.0
- **min_body_points**: 5.0, **min_range_points**: 10.0
- **Data file**: `new us30_UTC.csv` (env var `US30_DATA_DIR`)
- **Date splits**: train to 2025-10-31, validation/OOS to 2025-12-30

## Hardware Constraints (Apple M2 8GB)

- **Device**: MPS (Metal Performance Shaders)
- **Precision**: `float32` ONLY — never use `float64`
- **Memory cleanup**: Call `gc.collect()` and `torch.mps.empty_cache()` periodically
- **Batch sizes**: 128 (analyst), 512 (PPO agent)

## Configuration (`config/settings.py`)

Key dataclasses: `PathConfig`, `DataConfig`, `NormalizationConfig`, `DataSplitConfig`, `OODConfig`, `AnalystConfig`, `InstrumentConfig`, `TradingConfig`, `AgentConfig`, `FeatureConfig`, `BridgeConfig`.

Important parameters:
- `NormalizationConfig`: `rolling_window_size=5760`, `clip_value=5.0`, `min_rolling_samples=100`
- `TradingConfig`: `use_analyst=False`, `spread_pips=10.0`, `reward_scaling=0.01`, `sl_atr_multiplier=2.0`, `tp_atr_multiplier=6.0`
- `AgentConfig`: `n_envs=8`, `batch_size=512`, `total_timesteps=1B`
- `DataSplitConfig`: Date-based splits preferred (`train_end_date`, `validation_end_date`)
- Deprecated fields: `holding_bonus`, `early_exit_penalty`, `use_sparse_rewards`, `alpha_baseline_exposure`

## Coding Conventions

- Type hints mandatory; use `logging.getLogger(__name__)` for logging
- Configs as dataclasses matching `config/settings.py` patterns
- Files named by role: `train_*.py`, `*_env.py`, `*_agent.py`

## Commit Style

Conventional prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `config:`
Include before/after metrics (win rate, Sharpe, PnL) when available.

## Artifacts (gitignored)

- `data/processed/` — Parquet files, normalizers
- `models/analyst/` — `best_model.pt`, `normalizer_5m.pkl`, `normalizer_15m.pkl`, `normalizer_45m.pkl`
- `models/agent/` — `final_model.zip`, `market_feat_stats.npz`, `training_baseline.json`
- `results/` — Backtest CSVs, equity curves, trade logs
