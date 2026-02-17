# Gemini Context: US30 Hybrid AI Trading System

This file serves as a context guide for Gemini when working on this project.

## Project Overview

This is an end-to-end **Hybrid AI Trading System** specialized for **US30**. It combines a supervised "Market Analyst" model with a reinforcement learning "Sniper Agent" to make trading decisions. The system supports full training pipelines, out-of-sample backtesting, and live trading via a bridge to MetaTrader 5 (MT5).

### Core Architecture

The system operates as a two-stage decision pipeline:

1.  **Market Analyst (Supervised Learning)**:
    *   **Input**: Multi-timeframe market features (5m, 15m, 45m).
    *   **Model**: TCN (Temporal Convolutional Network) or Transformer (configurable).
    *   **Output**: A frozen "context vector" representing the market state.
2.  **Sniper Agent (Reinforcement Learning)**:
    *   **Input**: Analyst's context vector + raw market features + account state.
    *   **Model**: PPO (Proximal Policy Optimization) via `stable-baselines3`.
    *   **Output**: Trading actions (BUY, SELL, HOLD, Position Size).
3.  **Execution**:
    *   **Backtest**: Custom event-driven engine in `src/evaluation/backtest.py`.
    *   **Live**: TCP bridge (`src/live/mt5_bridge.py`) connecting to an MT5 Expert Advisor (`ClaudeConnector.mq5`).

## Key Directories & Files

| Path | Description |
| :--- | :--- |
| **`config/settings.py`** | **Critical**. Central configuration (paths, model params, trading rules) using dataclasses. |
| **`scripts/core/`** | Main entry points: `run_pipeline.py` (train/backtest), `run_mt5_bridge.py` (live). |
| **`src/data/`** | Data pipeline: `loader.py`, `resampler.py`, `features.py`, `normalizer.py`. |
| **`src/environments/`** | `trading_env.py`: Gymnasium environment for RL training. |
| **`src/models/`** | Neural network architectures (Analyst TCN/Transformer). |
| **`src/agents/`** | RL agent wrappers (`sniper_agent.py`, `recurrent_agent.py`). |
| **`src/live/`** | MT5 Bridge logic and neural broadcaster. |
| **`CLAUDE.md`** | Quick reference for developer commands and architectural notes. |
| **`requirements.txt`** | Python dependencies (PyTorch, Stable-Baselines3, etc.). |

## Common Workflows & Commands

**1. Setup**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Full Pipeline (Train -> Backtest)**
Runs the entire sequence: Data Loading -> Feature Engineering -> Analyst Training -> Agent Training -> Backtest.
```bash
python scripts/core/run_pipeline.py
```
*   *Options*: `--skip-analyst` (use existing), `--analyst-only`, `--backtest-only`.

**3. Live Trading Bridge**
Starts the Python server that listens for MT5 connections.
```bash
# Requires trained artifacts in models/
python scripts/core/run_mt5_bridge.py --host 127.0.0.1 --port 5555 --symbol US30
```

**4. Testing**
```bash
pytest tests/
pytest tests/test_parity.py -v  # Verify training vs. live bridge calculation parity
```

## Development Constraints & Conventions

*   **Hardware Optimization**: Code is optimized for Apple Silicon (M2). Defaults to `mps` device and `float32` precision. Avoid `float64`.
*   **Normalization**:
    *   **Training**: Computed on training split only to prevent data leakage.
    *   **Live/Backtest**: Uses rolling window normalization to match training statistics.
*   **Configuration**: All hyperparameters should be defined in `config/settings.py`, not hardcoded.
*   **Data Flow**: 1-minute OHLC CSV -> Resampled (5m/15m/45m) -> Feature Engineering -> Z-Score Normalization.
