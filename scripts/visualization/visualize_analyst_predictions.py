#!/usr/bin/env python3
"""
Visualize TCN Analyst predictions on a candlestick chart.

Loads the trained analyst model (best.pt) and pre-processed normalized data,
runs inference across the full dataset, and renders an interactive mplfinance
chart with probability overlays, signal markers, and a bullish/bearish heatmap.

Usage:
    python scripts/visualization/visualize_analyst_predictions.py
    python scripts/visualization/visualize_analyst_predictions.py --start 2025-11-01 --end 2025-12-30
    python scripts/visualization/visualize_analyst_predictions.py --threshold 0.7 --save
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplfinance as mpf

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn_analyst import load_tcn_analyst
from src.live.bridge_constants import MODEL_FEATURE_COLS

import gc
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_normalized_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-computed normalized parquet files."""
    data_dir = PROJECT_ROOT / "data" / "processed"

    df_5m = pd.read_parquet(data_dir / "features_5m_normalized.parquet")
    df_15m = pd.read_parquet(data_dir / "features_15m_normalized.parquet")
    df_45m = pd.read_parquet(data_dir / "features_45m_normalized.parquet")

    logger.info(f"Loaded normalized data: 5m={len(df_5m):,}, 15m={len(df_15m):,}, 45m={len(df_45m):,}")
    return df_5m, df_15m, df_45m


def run_inference(
    model,
    features_5m: np.ndarray,
    features_15m: np.ndarray,
    features_45m: np.ndarray,
    n_samples: int,
    start_idx: int,
    device: torch.device,
    lookback_5m: int = 48,
    lookback_15m: int = 16,
    lookback_45m: int = 6,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run sliding-window inference matching MultiTimeframeDataset logic."""

    subsample_15m = 3
    subsample_45m = 9

    p_up_all = np.full(n_samples, np.nan, dtype=np.float32)
    p_down_all = np.full(n_samples, np.nan, dtype=np.float32)
    edge_all = np.full(n_samples, np.nan, dtype=np.float32)
    confidence_all = np.full(n_samples, np.nan, dtype=np.float32)

    valid_indices = list(range(start_idx, n_samples))
    total_batches = (len(valid_indices) + batch_size - 1) // batch_size

    logger.info(f"Running inference on {len(valid_indices):,} bars in {total_batches} batches...")

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(valid_indices))
        batch_indices = valid_indices[batch_start:batch_end]

        # Build batch tensors
        x_5m_batch = []
        x_15m_batch = []
        x_45m_batch = []

        for idx in batch_indices:
            # 5m: direct lookback
            x_5m = features_5m[idx - lookback_5m + 1 : idx + 1]

            # 15m: subsample every 3rd bar (matching MultiTimeframeDataset)
            idx_range_15m = range(
                idx - (lookback_15m - 1) * subsample_15m,
                idx + 1,
                subsample_15m,
            )
            x_15m = features_15m[list(idx_range_15m)]

            # 45m: subsample every 9th bar
            idx_range_45m = range(
                idx - (lookback_45m - 1) * subsample_45m,
                idx + 1,
                subsample_45m,
            )
            x_45m = features_45m[list(idx_range_45m)]

            x_5m_batch.append(x_5m)
            x_15m_batch.append(x_15m)
            x_45m_batch.append(x_45m)

        # Stack: shape [batch, lookback, features] — model handles transpose internally
        x_5m_t = torch.tensor(np.array(x_5m_batch), dtype=torch.float32, device=device)
        x_15m_t = torch.tensor(np.array(x_15m_batch), dtype=torch.float32, device=device)
        x_45m_t = torch.tensor(np.array(x_45m_batch), dtype=torch.float32, device=device)

        with torch.no_grad():
            _, probs, _ = model.get_probabilities(x_5m_t, x_15m_t, x_45m_t)

        probs_np = probs.cpu().numpy()  # [batch, 2] = [p_down, p_up]

        for i, idx in enumerate(batch_indices):
            p_down_all[idx] = probs_np[i, 0]
            p_up_all[idx] = probs_np[i, 1]
            edge_all[idx] = probs_np[i, 1] - probs_np[i, 0]
            confidence_all[idx] = max(probs_np[i, 0], probs_np[i, 1])

        if (batch_num + 1) % 50 == 0 or batch_num == total_batches - 1:
            logger.info(f"  Batch {batch_num + 1}/{total_batches}")

        # Memory cleanup
        del x_5m_t, x_15m_t, x_45m_t, probs
        if batch_num % 20 == 0:
            gc.collect()
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass

    return p_up_all, p_down_all, edge_all, confidence_all


def plot_analyst_chart(
    df_ohlc: pd.DataFrame,
    p_up: np.ndarray,
    p_down: np.ndarray,
    edge: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.65,
    save_path: str | None = None,
    title: str = "US30 — TCN Analyst Predictions",
):
    """Render the multi-panel analyst visualization."""

    n = len(df_ohlc)

    # --- Signal markers (arrows where confidence > threshold) ---
    buy_signals = [np.nan] * n
    sell_signals = [np.nan] * n
    has_buy = False
    has_sell = False

    for i in range(n):
        if np.isnan(edge[i]):
            continue
        if confidence[i] >= threshold:
            if edge[i] > 0:
                buy_signals[i] = df_ohlc["Low"].iloc[i] * 0.9995
                has_buy = True
            else:
                sell_signals[i] = df_ohlc["High"].iloc[i] * 1.0005
                has_sell = True

    # --- Background heatmap via vlines coloring ---
    # Create edge color array for candlestick background
    edge_colors = []
    for i in range(n):
        if np.isnan(edge[i]):
            edge_colors.append((0.1, 0.1, 0.1, 0.0))  # transparent
        else:
            val = np.clip(edge[i], -1, 1)
            if val > 0:
                alpha = min(abs(val) * 0.6, 0.4)
                edge_colors.append((0.0, 0.8, 0.2, alpha))  # green
            else:
                alpha = min(abs(val) * 0.6, 0.4)
                edge_colors.append((0.9, 0.1, 0.1, alpha))  # red

    # --- Build addplot panels ---
    addplots = []

    # Panel 0 (main): Buy/Sell signal markers (only add if we have signals)
    if has_buy:
        addplots.append(
            mpf.make_addplot(
                buy_signals,
                type="scatter",
                markersize=30,
                marker="^",
                color="#00FF88",
                panel=0,
            )
        )
    if has_sell:
        addplots.append(
            mpf.make_addplot(
                sell_signals,
                type="scatter",
                markersize=30,
                marker="v",
                color="#FF4444",
                panel=0,
            )
        )

    # Panel 1: Probabilities
    p_up_series = pd.Series(p_up, index=df_ohlc.index)
    p_down_series = pd.Series(p_down, index=df_ohlc.index)

    addplots.append(
        mpf.make_addplot(
            p_up_series, panel=1, color="#00FF88", ylabel="Probability", width=0.8
        )
    )
    addplots.append(
        mpf.make_addplot(
            p_down_series, panel=1, color="#FF4444", width=0.8
        )
    )

    # Panel 2: Edge (directional signal)
    edge_series = pd.Series(edge, index=df_ohlc.index)
    addplots.append(
        mpf.make_addplot(
            edge_series,
            panel=2,
            color="#F39C12",
            ylabel="Edge",
            width=0.8,
            type="line",
        )
    )
    # Zero line for edge panel
    zero_line = pd.Series(0.0, index=df_ohlc.index)
    addplots.append(
        mpf.make_addplot(
            zero_line,
            panel=2,
            color="gray",
            width=0.5,
            linestyle="--",
        )
    )

    # Panel 3: Confidence
    conf_series = pd.Series(confidence, index=df_ohlc.index)
    addplots.append(
        mpf.make_addplot(
            conf_series,
            panel=3,
            color="#2E86AB",
            ylabel="Confidence",
            width=0.8,
            type="line",
        )
    )
    # Threshold line
    thresh_line = pd.Series(threshold, index=df_ohlc.index)
    addplots.append(
        mpf.make_addplot(
            thresh_line,
            panel=3,
            color="gray",
            width=0.5,
            linestyle="--",
        )
    )

    # --- Style ---
    mc = mpf.make_marketcolors(
        up="#26A69A",
        down="#EF5350",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#1a1a1a",
        edgecolor="#333333",
        gridcolor="#2a2a2a",
        gridstyle="--",
        gridaxis="both",
        y_on_right=True,
        rc={
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        },
    )

    # --- Render ---
    panel_ratios = (4, 1, 1, 1)
    fig_kwargs = dict(
        type="candle",
        style=style,
        addplot=addplots,
        title=title,
        panel_ratios=panel_ratios,
        figsize=(20, 12),
        tight_layout=True,
        warn_too_much_data=1_000_000,
    )

    fig, axes = mpf.plot(df_ohlc, **fig_kwargs, returnfig=True)

    # Add background heatmap to main panel via colored vlines
    ax_main = axes[0]
    for i in range(n):
        if edge_colors[i][3] > 0.01:  # skip transparent
            ax_main.axvspan(
                i - 0.5, i + 0.5,
                facecolor=edge_colors[i][:3],
                alpha=edge_colors[i][3],
                zorder=0,
            )

    # Legend for probability panel (panel 1 = axes[2] since each panel has 2 axes)
    axes[2].legend(["p_up", "p_down"], loc="upper left", fontsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Chart saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize TCN Analyst predictions on US30 chart")
    parser.add_argument("--model", type=str, default=None, help="Path to analyst model (default: models/analyst/best.pt)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold for signal markers (default: 0.65)")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size (default: 512)")
    parser.add_argument("--save", type=str, default=None, help="Save chart to file instead of showing (e.g. analyst_chart.png)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load data
    df_5m, df_15m, df_45m = load_normalized_data()

    # Filter date range
    if args.start:
        start = pd.Timestamp(args.start)
        df_5m = df_5m[df_5m.index >= start]
        df_15m = df_15m[df_15m.index >= start]
        df_45m = df_45m[df_45m.index >= start]
    if args.end:
        end = pd.Timestamp(args.end)
        df_5m = df_5m[df_5m.index <= end]
        df_15m = df_15m[df_15m.index <= end]
        df_45m = df_45m[df_45m.index <= end]

    # Align indices (15m/45m are forward-filled onto 5m index in the parquets)
    common_idx = df_5m.index.intersection(df_15m.index).intersection(df_45m.index)
    df_5m = df_5m.loc[common_idx]
    df_15m = df_15m.loc[common_idx]
    df_45m = df_45m.loc[common_idx]
    logger.info(f"Aligned data: {len(df_5m):,} bars")

    # Extract feature arrays
    feature_cols = [c for c in MODEL_FEATURE_COLS if c in df_5m.columns]
    features_5m = df_5m[feature_cols].values.astype(np.float32)
    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_45m = df_45m[feature_cols].values.astype(np.float32)

    n_features = len(feature_cols)
    logger.info(f"Features: {n_features} columns")

    # Load model
    model_path = args.model or str(PROJECT_ROOT / "models" / "analyst" / "best.pt")
    feature_dims = {"5m": n_features, "15m": n_features, "45m": n_features}
    model = load_tcn_analyst(model_path, feature_dims, device=device, freeze=True)
    logger.info(f"Loaded analyst from {model_path}")

    # Compute start index (same as MultiTimeframeDataset)
    lookback_5m, lookback_15m, lookback_45m = 48, 16, 6
    start_idx = max(lookback_5m, lookback_15m * 3, lookback_45m * 9)

    # Run inference
    p_up, p_down, edge, confidence = run_inference(
        model=model,
        features_5m=features_5m,
        features_15m=features_15m,
        features_45m=features_45m,
        n_samples=len(df_5m),
        start_idx=start_idx,
        device=device,
        lookback_5m=lookback_5m,
        lookback_15m=lookback_15m,
        lookback_45m=lookback_45m,
        batch_size=args.batch_size,
    )

    # Free model memory
    del model
    gc.collect()
    try:
        torch.mps.empty_cache()
    except AttributeError:
        pass

    # Build OHLC DataFrame for mplfinance
    ohlc_cols = ["open", "high", "low", "close"]
    df_ohlc = df_5m[ohlc_cols].copy()
    df_ohlc.columns = ["Open", "High", "Low", "Close"]

    # Stats
    valid_mask = ~np.isnan(p_up)
    n_valid = valid_mask.sum()
    n_bullish = ((edge > 0) & valid_mask).sum()
    n_bearish = ((edge < 0) & valid_mask).sum()
    n_signals = ((confidence >= args.threshold) & valid_mask).sum()
    logger.info(f"Predictions: {n_valid:,} bars | Bullish: {n_bullish:,} | Bearish: {n_bearish:,}")
    logger.info(f"Signals (conf >= {args.threshold}): {n_signals:,} ({n_signals / max(n_valid, 1) * 100:.1f}%)")
    logger.info(f"Avg confidence: {np.nanmean(confidence):.3f} | Avg |edge|: {np.nanmean(np.abs(edge)):.3f}")

    # Plot
    date_range = f" ({args.start or 'start'} to {args.end or 'end'})" if args.start or args.end else ""
    title = f"US30 — TCN Analyst Predictions{date_range}  |  threshold={args.threshold}"

    plot_analyst_chart(
        df_ohlc=df_ohlc,
        p_up=p_up,
        p_down=p_down,
        edge=edge,
        confidence=confidence,
        threshold=args.threshold,
        save_path=args.save,
        title=title,
    )


if __name__ == "__main__":
    main()
