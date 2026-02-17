#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CheckpointRow:
    checkpoint: str
    checkpoint_abs: str
    steps: int
    total_return_pct: float
    max_drawdown_pct: float
    calmar_ratio: float


@dataclass(frozen=True)
class Zone:
    name: str
    start_idx: int
    end_idx_excl: int
    return_floor_pct: float | None
    dd_ceiling_pct: float | None
    color: str
    alpha: float


def _load_success_rows(path: Path) -> list[CheckpointRow]:
    rows: list[CheckpointRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line %s in %s", line_num, path)
                continue
            if payload.get("type") != "success":
                continue
            data: dict[str, Any] = payload.get("data") or {}
            try:
                rows.append(
                    CheckpointRow(
                        checkpoint=str(data["checkpoint"]),
                        checkpoint_abs=str(data.get("checkpoint_abs", "")),
                        steps=int(data["steps"]),
                        total_return_pct=float(data["total_return_pct"]),
                        max_drawdown_pct=float(data["max_drawdown_pct"]),
                        calmar_ratio=float(data["calmar_ratio"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed row at line %s (%s): %s", line_num, exc, data)
                continue

    rows.sort(key=lambda r: r.steps)
    return rows


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    if mask.size == 0:
        return None

    best_start = -1
    best_end_excl = -1
    current_start: int | None = None

    for i, ok in enumerate(mask.tolist()):
        if ok and current_start is None:
            current_start = i
        if (not ok or i == mask.size - 1) and current_start is not None:
            end_excl = i if not ok else i + 1
            if end_excl - current_start > best_end_excl - best_start:
                best_start = current_start
                best_end_excl = end_excl
            current_start = None

    if best_start < 0:
        return None
    return best_start, best_end_excl


def _pick_dd_first_recommendation(rows: list[CheckpointRow], start: int, end_excl: int) -> int:
    segment = rows[start:end_excl]
    # Primary: minimize drawdown; Secondary: maximize Calmar; Tertiary: maximize return.
    best_local_idx = min(
        range(len(segment)),
        key=lambda i: (
            segment[i].max_drawdown_pct,
            -segment[i].calmar_ratio,
            -segment[i].total_return_pct,
            segment[i].steps,
        ),
    )
    return start + best_local_idx


def _millions(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) / 1e6


def _fmt_steps_m(steps: int) -> str:
    return f"{steps / 1e6:.1f}M"


def _setup_style() -> None:
    preferred = "seaborn-v0_8-darkgrid"
    plt.style.use(preferred if preferred in plt.style.available else "bmh")


def _span_bounds(steps_m: np.ndarray, start_idx: int, end_idx_excl: int) -> tuple[float, float]:
    start = float(steps_m[start_idx])
    end = float(steps_m[end_idx_excl - 1])
    if steps_m.size >= 2:
        delta = float(np.median(np.diff(steps_m)))
    else:
        delta = 0.1
    return start - delta / 2.0, end + delta / 2.0


def _zone_center_m(steps: np.ndarray, start_idx: int, end_idx_excl: int) -> int:
    start = float(steps[start_idx])
    end = float(steps[end_idx_excl - 1])
    return int(round(((start + end) / 2.0) / 1e6))


def _plot_full_run(
    ax: plt.Axes,
    *,
    steps_m: np.ndarray,
    returns: np.ndarray,
    zones: list[Zone],
    stable_return_floor_pct: float,
    stable_rec_idx: int,
    stable_rec_steps: int,
) -> None:
    ax.plot(steps_m, returns, color="royalblue", linewidth=1.0, alpha=0.9)
    ax.fill_between(steps_m, returns, 0.0, color="royalblue", alpha=0.15)

    ax.axhline(
        stable_return_floor_pct,
        color="orange",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label=f"{stable_return_floor_pct:.0f}% Return threshold",
    )

    for zone in zones:
        x0, x1 = _span_bounds(steps_m, zone.start_idx, zone.end_idx_excl)
        ax.axvspan(x0, x1, color=zone.color, alpha=zone.alpha, label=zone.name)

    ax.axvline(steps_m[stable_rec_idx], color="red", linewidth=1.2, alpha=0.9)
    ax.scatter(
        [steps_m[stable_rec_idx]],
        [returns[stable_rec_idx]],
        color="red",
        s=80,
        marker="*",
        zorder=5,
        label=f"{_fmt_steps_m(stable_rec_steps)} (Stable pick)",
    )

    ax.set_title("Full Training Run - Stable Zones Highlighted", fontsize=14, weight="bold")
    ax.set_ylabel("Total Return %")
    ax.set_xlabel("Steps (Millions)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def _plot_zoom_zone(
    ax: plt.Axes,
    *,
    title: str,
    steps_m: np.ndarray,
    returns: np.ndarray,
    zone: Zone,
    rec_idx: int,
    rec_steps: int,
    rec_label: str,
    floor_label: str,
    margin_m: float,
) -> None:
    zone_x0, zone_x1 = _span_bounds(steps_m, zone.start_idx, zone.end_idx_excl)
    xmin = max(float(steps_m.min()), zone_x0 - margin_m)
    xmax = min(float(steps_m.max()), zone_x1 + margin_m)
    subset = (steps_m >= xmin) & (steps_m <= xmax)

    ax.plot(
        steps_m[subset],
        returns[subset],
        color="blue",
        linewidth=1.2,
        marker="o",
        markersize=3.5,
        alpha=0.9,
    )

    if zone.return_floor_pct is not None:
        ax.axhline(
            zone.return_floor_pct,
            color="green" if zone.name.startswith("Perf") else "orange",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label=floor_label,
        )

    ax.axvspan(zone_x0, zone_x1, color=zone.color, alpha=0.18, label="Stable plateau")
    ax.axvline(
        steps_m[rec_idx],
        color="red",
        linewidth=1.3,
        alpha=0.9,
        label=f"{_fmt_steps_m(rec_steps)} ({rec_label})",
    )

    ax.scatter(
        [steps_m[rec_idx]],
        [returns[rec_idx]],
        color="red",
        s=110,
        marker="*",
        zorder=6,
    )

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel("Total Return %")
    ax.set_xlabel("Steps (Millions)")
    ax.set_xlim(xmin, xmax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze checkpoint stability (DD-prioritized) and plot stable zones.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/finetune_oos_ranking.jsonl",
        help="Path to JSONL checkpoint results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/checkpoint_stability_analysis.png",
        help="Output PNG path",
    )
    parser.add_argument("--stable-return-floor", type=float, default=40.0, help="Stable zone min return threshold (%)")
    parser.add_argument("--stable-dd-ceiling", type=float, default=20.0, help="Stable zone max drawdown ceiling (%)")
    parser.add_argument("--wide-return-floor", type=float, default=35.0, help="Wide zone min return threshold (%)")
    parser.add_argument("--wide-dd-ceiling", type=float, default=30.0, help="Wide zone max drawdown ceiling (%)")
    parser.add_argument("--perf-return-floor", type=float, default=60.0, help="Performance plateau min return threshold (%)")
    parser.add_argument("--width-px", type=int, default=2382, help="Output image width (px)")
    parser.add_argument("--height-px", type=int, default=2080, help="Output image height (px)")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (INFO, WARNING, ...)")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    rows = _load_success_rows(input_path)
    if not rows:
        raise RuntimeError(f"No successful rows found in {input_path}")

    steps = np.array([r.steps for r in rows], dtype=np.int64)
    steps_m = _millions(steps)
    returns = np.array([r.total_return_pct for r in rows], dtype=np.float64)
    drawdowns = np.array([r.max_drawdown_pct for r in rows], dtype=np.float64)

    stable_mask = (returns >= float(args.stable_return_floor)) & (drawdowns <= float(args.stable_dd_ceiling))
    wide_mask = (returns >= float(args.wide_return_floor)) & (drawdowns <= float(args.wide_dd_ceiling))
    perf_mask = returns >= float(args.perf_return_floor)

    stable_run = _longest_true_run(stable_mask)
    wide_run = _longest_true_run(wide_mask)
    perf_run = _longest_true_run(perf_mask)

    if stable_run is None:
        raise RuntimeError("No stable DD-first zone found; try relaxing thresholds.")
    if wide_run is None:
        raise RuntimeError("No wide stable zone found; try relaxing thresholds.")
    if perf_run is None:
        logger.warning("No performance plateau found at %.1f%%; skipping.", float(args.perf_return_floor))

    stable_start, stable_end = stable_run
    wide_start, wide_end = wide_run

    stable_rec_idx = _pick_dd_first_recommendation(rows, stable_start, stable_end)
    wide_rec_idx = _pick_dd_first_recommendation(rows, wide_start, wide_end)

    stable_region_m = _zone_center_m(steps, stable_start, stable_end)
    wide_region_m = _zone_center_m(steps, wide_start, wide_end)

    perf_zone: Zone | None = None
    perf_rec_idx: int | None = None
    if perf_run is not None:
        perf_start, perf_end = perf_run
        perf_region_m = _zone_center_m(steps, perf_start, perf_end)
        perf_zone = Zone(
            name=f"Perf zone ({perf_region_m}M)",
            start_idx=perf_start,
            end_idx_excl=perf_end,
            return_floor_pct=float(args.perf_return_floor),
            dd_ceiling_pct=None,
            color="#2ca02c",
            alpha=0.18,
        )
        perf_rec_idx = _pick_dd_first_recommendation(rows, perf_start, perf_end)

    zones: list[Zone] = [
        Zone(
            name=f"Stable DD-first zone ({stable_region_m}M)",
            start_idx=stable_start,
            end_idx_excl=stable_end,
            return_floor_pct=float(args.stable_return_floor),
            dd_ceiling_pct=float(args.stable_dd_ceiling),
            color="#ff9896",
            alpha=0.18,
        ),
        Zone(
            name=f"Wide stable zone ({wide_region_m}M)",
            start_idx=wide_start,
            end_idx_excl=wide_end,
            return_floor_pct=float(args.wide_return_floor),
            dd_ceiling_pct=float(args.wide_dd_ceiling),
            color="#9467bd",
            alpha=0.18,
        ),
    ]
    if perf_zone is not None:
        zones.append(perf_zone)

    _setup_style()
    fig_w = float(args.width_px) / float(args.dpi)
    fig_h = float(args.height_px) / float(args.dpi)
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=int(args.dpi))

    _plot_full_run(
        axes[0],
        steps_m=steps_m,
        returns=returns,
        zones=zones,
        stable_return_floor_pct=float(args.stable_return_floor),
        stable_rec_idx=stable_rec_idx,
        stable_rec_steps=rows[stable_rec_idx].steps,
    )

    stable_title = f"{stable_region_m}M Region - DD-First Stable Plateau"
    _plot_zoom_zone(
        axes[1],
        title=stable_title,
        steps_m=steps_m,
        returns=returns,
        zone=zones[0],
        rec_idx=stable_rec_idx,
        rec_steps=rows[stable_rec_idx].steps,
        rec_label="Recommended",
        floor_label=f"{args.stable_return_floor:.0f}% floor",
        margin_m=0.9,
    )

    wide_title = f"{wide_region_m}M Region - Widest Stable Zone"
    _plot_zoom_zone(
        axes[2],
        title=wide_title,
        steps_m=steps_m,
        returns=returns,
        zone=zones[1],
        rec_idx=wide_rec_idx,
        rec_steps=rows[wide_rec_idx].steps,
        rec_label="Safest",
        floor_label=f"{args.wide_return_floor:.0f}% floor",
        margin_m=1.2,
    )

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    rec = rows[stable_rec_idx]
    stable_zone_start = rows[stable_start].steps
    stable_zone_end = rows[stable_end - 1].steps
    logger.info(
        "Stable DD-first zone: %s → %s (%d checkpoints)",
        _fmt_steps_m(stable_zone_start),
        _fmt_steps_m(stable_zone_end),
        stable_end - stable_start,
    )
    logger.info(
        "Recommended checkpoint: %s (return=%.2f%%, maxDD=%.2f%%, calmar=%.2f) -> %s",
        _fmt_steps_m(rec.steps),
        rec.total_return_pct,
        rec.max_drawdown_pct,
        rec.calmar_ratio,
        rec.checkpoint,
    )
    if perf_zone is not None and perf_rec_idx is not None:
        perf_rec = rows[perf_rec_idx]
        logger.info(
            "Performance plateau: %s → %s; best DD-first pick=%s (return=%.2f%%, maxDD=%.2f%%)",
            _fmt_steps_m(rows[perf_zone.start_idx].steps),
            _fmt_steps_m(rows[perf_zone.end_idx_excl - 1].steps),
            _fmt_steps_m(perf_rec.steps),
            perf_rec.total_return_pct,
            perf_rec.max_drawdown_pct,
        )
    logger.info("Plot saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
