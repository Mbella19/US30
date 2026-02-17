# Visualization Scripts

## Naming Convention
- `plot_*.py` - Single static matplotlib plots (PNG output)
- `visualize_*.py` - Multi-panel or complex visualizations

## Script Purposes

| Script | Purpose | Notes |
|--------|---------|-------|
| `plot_evolution.py` | Checkpoint performance dashboard | Contains hardcoded data - update for new runs |
| `plot_logs.py` | Parse and plot training logs | Accepts log file as positional arg |
| `plot_monthly_heatmap.py` | Monthly returns heatmap | Reads from results/ directory |
| `plot_ppo_stability.py` | PPO training health metrics | Auto-finds latest log dir |
| `plot_specific_equity.py` | Equity curve for specific result | Reads from results/ directory |
| `plot_sr_levels.py` | Support/Resistance visualization | Requires OHLC data |
| `plot_training_progress.py` | Training progress charts | **UPDATE**: hardcoded log path |
| `plot_winning_trades.py` | Candlestick charts of winners | Requires trades data |
| `visualize_backtest.py` | Equity curve + drawdown | Accepts results_dir as arg |
| `visualize_champion_trades.py` | Best/worst trade charts | Accepts results_dir, data_path |
| `visualize_continuous_trades.py` | Sequential trade plots | Accepts multiple positional args |
| `visualize_new_trades.py` | Trade visualization variant | Similar to continuous_trades |
| `visualize_structure.py` | Market structure visualization | **UPDATE**: hardcoded data path |

## Scripts with Hardcoded Paths

The following scripts have hardcoded paths that may need updating:

1. **`plot_training_progress.py`** (line 146)
   - Update: `log_file = "models/agent/training_*.log"`
   - Find logs: `ls models/agent/training_*.log`

2. **`plot_evolution.py`** (line 22-40)
   - Contains hardcoded checkpoint performance data
   - Update the `data[]` array with metrics from your training runs

3. **`visualize_structure.py`** (line 24)
   - Update: `df = pd.read_parquet('data/processed/features_15m.parquet')`
   - Change to your processed features file

## Usage Examples

```bash
# Plot training logs
python scripts/visualization/plot_logs.py models/agent/training.log

# Visualize backtest results
python scripts/visualization/visualize_backtest.py results/latest/

# PPO stability check (auto-finds latest)
python scripts/visualization/plot_ppo_stability.py

# Evolution dashboard (uses hardcoded data)
python scripts/visualization/plot_evolution.py
```
