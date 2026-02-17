#!/usr/bin/env python3
"""
Visualize an SB3 PPO checkpoint (weights + policy behavior).

Outputs (in --out-dir):
- weight_distributions.png: histograms for actor/critic/head parameters
- param_stats.json: per-parameter statistics (mean/std/norm/etc)
- input_segment_importance.png: first-layer input importance by observation segment
- policy_rollout.png: (optional) action probabilities over a short rollout
- action_frequencies.png: (optional) action counts over the rollout

Example:
  python scripts/visualization/visualize_agent_checkpoint.py \\
    --checkpoint models/agent/checkpoints/sniper_model_76500000_steps.zip \\
    --rollout-steps 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configure matplotlib cache dirs BEFORE importing matplotlib.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from config.settings import Config
from src.live.bridge_constants import MODEL_FEATURE_COLS, MARKET_FEATURE_COLS
from src.models.analyst import load_analyst
from src.training.train_agent import create_trading_env, prepare_env_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObsSegment:
    name: str
    start: int
    end: int  # exclusive

    @property
    def size(self) -> int:
        return self.end - self.start


def _setup_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _tensor_to_flat_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(dtype=torch.float32).numpy().ravel()


def _param_stats(name: str, t: torch.Tensor) -> dict[str, Any]:
    vals = _tensor_to_flat_np(t)
    if vals.size == 0:
        return {"name": name, "shape": list(t.shape), "numel": 0}

    l2 = float(np.linalg.norm(vals))
    l2_rms = float(l2 / np.sqrt(vals.size))
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "abs_mean": float(np.mean(np.abs(vals))),
        "abs_max": float(np.max(np.abs(vals))),
        "l2_norm": l2,
        "l2_rms": l2_rms,
        "zero_frac": float(np.mean(vals == 0.0)),
    }


def _group_params(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {
        "actor_mlp": [],
        "critic_mlp": [],
        "action_head": [],
        "value_head": [],
        "other": [],
    }
    for name, t in state_dict.items():
        vals = _tensor_to_flat_np(t)
        if name.startswith("mlp_extractor.policy_net."):
            buckets["actor_mlp"].append(vals)
        elif name.startswith("mlp_extractor.value_net."):
            buckets["critic_mlp"].append(vals)
        elif name.startswith("action_net."):
            buckets["action_head"].append(vals)
        elif name.startswith("value_net."):
            buckets["value_head"].append(vals)
        else:
            buckets["other"].append(vals)

    grouped: dict[str, np.ndarray] = {}
    for k, parts in buckets.items():
        if not parts:
            continue
        grouped[k] = np.concatenate(parts).astype(np.float32, copy=False)
    return grouped


def _plot_weight_distributions(
    grouped: dict[str, np.ndarray],
    *,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "bmh")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    keys = ["actor_mlp", "critic_mlp", "action_head", "value_head"]
    for ax, key in zip(axes.flatten(), keys, strict=False):
        vals = grouped.get(key)
        if vals is None or vals.size == 0:
            ax.text(0.5, 0.5, f"{key}\n(no params)", ha="center", va="center")
            ax.set_axis_off()
            continue

        p_low, p_high = np.quantile(vals, [0.005, 0.995])
        ax.hist(vals, bins=120, color="#2E86AB", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.6)
        if p_low < p_high:
            ax.set_xlim(float(p_low), float(p_high))
        ax.set_title(f"{key} (n={vals.size:,})")
        ax.set_xlabel("weight value")
        ax.set_ylabel("count")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_param_norms(
    stats: list[dict[str, Any]],
    *,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [s for s in stats if "l2_norm" in s]
    rows.sort(key=lambda r: float(r["l2_norm"]), reverse=True)

    names = [r["name"] for r in rows]
    norms = [float(r["l2_norm"]) for r in rows]

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "bmh")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(names))))
    ax.barh(names[::-1], norms[::-1], color="#E94F37", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("L2 norm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _infer_obs_segments(
    *,
    obs_dim: int,
    context_dim: int,
    agent_lookback_window: int,
    analyst_metrics_dim: int,
    n_market_features: int,
) -> list[ObsSegment]:
    segments: list[ObsSegment] = []
    idx = 0

    if context_dim > 0:
        segments.append(ObsSegment("context", idx, idx + context_dim))
        idx += context_dim

    segments.append(ObsSegment("position_state", idx, idx + 4))
    idx += 4

    segments.append(ObsSegment("market_features", idx, idx + n_market_features))
    idx += n_market_features

    if analyst_metrics_dim > 0:
        segments.append(ObsSegment("analyst_metrics", idx, idx + analyst_metrics_dim))
        idx += analyst_metrics_dim

    segments.append(ObsSegment("sl_tp", idx, idx + 2))
    idx += 2

    segments.append(ObsSegment("hold_features", idx, idx + 4))
    idx += 4

    if agent_lookback_window > 0:
        segments.append(ObsSegment("returns_window", idx, idx + agent_lookback_window))
        idx += agent_lookback_window

    if idx != obs_dim:
        raise ValueError(f"Segment definition mismatch: built={idx}, expected obs_dim={obs_dim}")

    return segments


def _plot_input_segment_importance(
    state_dict: dict[str, torch.Tensor],
    *,
    obs_segments: list[ObsSegment],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actor_w = state_dict.get("mlp_extractor.policy_net.0.weight")
    critic_w = state_dict.get("mlp_extractor.value_net.0.weight")
    if actor_w is None or critic_w is None:
        raise KeyError("Missing first-layer weights in policy state_dict")

    actor_imp = np.mean(np.abs(actor_w.detach().cpu().numpy().astype(np.float32)), axis=0)
    critic_imp = np.mean(np.abs(critic_w.detach().cpu().numpy().astype(np.float32)), axis=0)

    seg_names: list[str] = []
    actor_seg: list[float] = []
    critic_seg: list[float] = []
    for seg in obs_segments:
        seg_names.append(f"{seg.name}\n({seg.size})")
        actor_seg.append(float(actor_imp[seg.start:seg.end].mean()))
        critic_seg.append(float(critic_imp[seg.start:seg.end].mean()))

    x = np.arange(len(seg_names))
    width = 0.42

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "bmh")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, actor_seg, width, label="actor first-layer |w| mean", color="#2E86AB", alpha=0.85)
    ax.bar(x + width / 2, critic_seg, width, label="critic first-layer |w| mean", color="#E94F37", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_names, rotation=0)
    ax.set_title(title)
    ax.set_ylabel("mean(|w|) across segment inputs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_tail_parquet(path: Path, *, needed_rows: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if needed_rows <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= needed_rows:
        return df.copy()
    return df.iloc[-needed_rows:].copy()


def _build_rollout_env(
    checkpoint: PPO,
    *,
    cfg: Config,
    rollout_steps: int,
) -> tuple[Any, dict[str, Any]]:
    obs_dim = int(getattr(checkpoint.observation_space, "shape", (0,))[0])

    lookback_5m = int(cfg.analyst.lookback_5m)
    lookback_15m = int(cfg.analyst.lookback_15m)
    lookback_45m = int(cfg.analyst.lookback_45m)
    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    needed_rows = start_idx + rollout_steps

    p5 = cfg.paths.data_processed / "features_5m_normalized.parquet"
    p15 = cfg.paths.data_processed / "features_15m_normalized.parquet"
    p45 = cfg.paths.data_processed / "features_45m_normalized.parquet"
    for p in (p5, p15, p45):
        if not p.exists():
            raise FileNotFoundError(f"Missing processed features parquet: {p}")

    logger.info("Loading processed features (tail=%d rows)...", needed_rows)
    df_5m = _load_tail_parquet(p5, needed_rows=needed_rows)
    df_15m = _load_tail_parquet(p15, needed_rows=needed_rows)
    df_45m = _load_tail_parquet(p45, needed_rows=needed_rows)

    feature_cols = list(MODEL_FEATURE_COLS)
    data_5m, data_15m, data_45m, close_prices, market_features, returns, rolling_lookback_data = prepare_env_data(
        df_5m,
        df_15m,
        df_45m,
        feature_cols,
        lookback_5m=lookback_5m,
        lookback_15m=lookback_15m,
        lookback_45m=lookback_45m,
    )

    n_samples = len(close_prices)
    if n_samples <= 0:
        raise ValueError("Not enough rows to build rollout env; increase available data or reduce rollout_steps.")

    timestamps = None
    ohlc_data = None
    if isinstance(df_5m.index, pd.DatetimeIndex):
        timestamps = (df_5m.index[start_idx:start_idx + n_samples].astype("int64") // 10**9).values
    if all(c in df_5m.columns for c in ("open", "high", "low", "close")):
        ohlc_data = df_5m[["open", "high", "low", "close"]].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Compute normalization stats from the first 85% of this slice.
    train_end_idx = max(1, int(0.85 * len(market_features)))
    mkt_train = market_features[:train_end_idx]
    market_feat_mean = mkt_train.mean(axis=0).astype(np.float32)
    market_feat_std = mkt_train.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)

    analyst_path = cfg.paths.models_analyst / "best.pt"
    if not analyst_path.exists():
        raise FileNotFoundError(f"Analyst model not found: {analyst_path}")

    feature_dims = {"5m": len(feature_cols), "15m": len(feature_cols), "45m": len(feature_cols)}
    analyst = load_analyst(str(analyst_path), feature_dims, device=torch.device("cpu"), freeze=True)

    # Deterministic rollout: disable noise and ensure episode length >= requested steps.
    trading_cfg = cfg.trading
    trading_cfg.noise_level = 0.0
    trading_cfg.max_steps_per_episode = max(int(trading_cfg.max_steps_per_episode), int(rollout_steps))

    env = create_trading_env(
        data_5m=data_5m,
        data_15m=data_15m,
        data_45m=data_45m,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=analyst,
        config=trading_cfg,
        device=torch.device("cpu"),
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        regime_labels=None,
        use_regime_sampling=False,
        precomputed_analyst_cache=None,
        ohlc_data=ohlc_data,
        timestamps=timestamps,
        returns=returns,
        use_analyst=bool(trading_cfg.use_analyst),
        rolling_lookback_data=rolling_lookback_data,
    )

    # Basic compatibility check before rollout.
    if getattr(env.observation_space, "shape", None) != getattr(checkpoint.observation_space, "shape", None):
        raise ValueError(
            "Env/model observation space mismatch. "
            f"env={env.observation_space} model={checkpoint.observation_space}"
        )
    if str(env.action_space) != str(checkpoint.action_space):
        raise ValueError(f"Env/model action space mismatch. env={env.action_space} model={checkpoint.action_space}")

    meta = {
        "obs_dim": obs_dim,
        "context_dim": int(getattr(env, "effective_context_dim", 0)),
        "analyst_metrics_dim": int(getattr(env, "analyst_metrics_dim", 0)),
        "agent_lookback_window": int(getattr(env, "agent_lookback_window", 0)),
        "n_market_features": int(market_features.shape[1]),
        "market_feature_cols": [f"{tf}:{col}" for tf in ("5m", "15m", "45m") for col in MARKET_FEATURE_COLS],
    }

    return env, meta


def _extract_multidiscrete_probs(dist: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (direction_probs[3], size_probs[4]) for MultiDiscrete([3,4]).
    """
    if hasattr(dist, "distribution") and isinstance(dist.distribution, list) and len(dist.distribution) >= 2:
        dir_probs = dist.distribution[0].probs.detach().cpu().numpy()[0].astype(np.float32)
        size_probs = dist.distribution[1].probs.detach().cpu().numpy()[0].astype(np.float32)
        return dir_probs, size_probs

    if hasattr(dist, "distribution") and hasattr(dist.distribution, "logits"):
        logits = dist.distribution.logits.detach().cpu().numpy()[0].astype(np.float32)
        dir_logits = logits[:3]
        size_logits = logits[3:7]
        return _softmax(dir_logits), _softmax(size_logits)

    raise TypeError("Unsupported SB3 distribution structure for MultiDiscrete")


def _run_policy_rollout(
    model: PPO,
    env: Any,
    *,
    steps: int,
    deterministic: bool,
) -> pd.DataFrame:
    obs, _info = env.reset(seed=0)

    rows: list[dict[str, Any]] = []
    for t in range(int(steps)):
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            value = model.policy.predict_values(obs_tensor).detach().cpu().numpy()[0].astype(np.float32)

        dir_probs, size_probs = _extract_multidiscrete_probs(dist)
        action, _states = model.predict(obs, deterministic=deterministic)

        # Capture pre-step state for interpretability
        close = float(env.close_prices[env.current_idx])
        timestamp = None
        if getattr(env, "timestamps", None) is not None:
            timestamp = int(env.timestamps[env.current_idx])

        rows.append(
            {
                "t": int(t),
                "timestamp": timestamp,
                "close": close,
                "position": int(getattr(env, "position", 0)),
                "position_size": float(getattr(env, "position_size", 0.0)),
                "action_dir": int(action[0]),
                "action_size": int(action[1]),
                "p_flat": float(dir_probs[0]),
                "p_long": float(dir_probs[1]),
                "p_short": float(dir_probs[2]),
                "p_size_0": float(size_probs[0]),
                "p_size_1": float(size_probs[1]),
                "p_size_2": float(size_probs[2]),
                "p_size_3": float(size_probs[3]),
                "value": float(value.item() if hasattr(value, "item") else value),
            }
        )

        obs, reward, terminated, truncated, _info = env.step(action)
        rows[-1]["reward"] = float(reward)
        if terminated or truncated:
            break

    return pd.DataFrame(rows)


def _plot_policy_rollout(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError("Rollout dataframe is empty")

    x = df["t"].to_numpy()

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "bmh")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Price
    axes[0].plot(x, df["close"], color="black", linewidth=1.25)
    axes[0].set_ylabel("close")

    # Direction probs
    axes[1].plot(x, df["p_flat"], label="flat", color="#95A5A6")
    axes[1].plot(x, df["p_long"], label="long", color="#27AE60")
    axes[1].plot(x, df["p_short"], label="short", color="#C0392B")
    axes[1].set_ylabel("dir prob")
    axes[1].legend(loc="upper right", ncol=3)

    # Action + position
    axes[2].step(x, df["action_dir"], where="post", label="action_dir", color="#2E86AB")
    axes[2].step(x, df["position"], where="post", label="position", color="#F39C12", alpha=0.85)
    axes[2].set_ylabel("dir")
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["flat", "long", "short"])
    axes[2].legend(loc="upper right")

    # Size probs
    axes[3].plot(x, df["p_size_0"], label="size_0", alpha=0.9)
    axes[3].plot(x, df["p_size_1"], label="size_1", alpha=0.9)
    axes[3].plot(x, df["p_size_2"], label="size_2", alpha=0.9)
    axes[3].plot(x, df["p_size_3"], label="size_3", alpha=0.9)
    axes[3].set_ylabel("size prob")
    axes[3].legend(loc="upper right", ncol=4, fontsize=9)
    axes[3].set_xlabel("step")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_action_frequencies(df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError("Rollout dataframe is empty")

    dir_counts = df["action_dir"].value_counts().sort_index()
    size_counts = df["action_size"].value_counts().sort_index()

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "bmh")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    axes[0].bar(dir_counts.index.astype(int), dir_counts.values, color="#2E86AB", alpha=0.85)
    axes[0].set_title("Direction action counts")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["flat", "long", "short"])

    axes[1].bar(size_counts.index.astype(int), size_counts.values, color="#E94F37", alpha=0.85)
    axes[1].set_title("Size action counts")
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xticklabels(["0", "1", "2", "3"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize SB3 agent checkpoint weights and policy behavior.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SB3 .zip checkpoint (e.g. models/agent/checkpoints/sniper_model_76500000_steps.zip)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: results/model_viz/<checkpoint_stem>/)",
    )
    parser.add_argument("--no-rollout", action="store_true", help="Skip policy rollout visualizations.")
    parser.add_argument("--rollout-steps", type=int, default=2000, help="Max rollout steps (default: 2000).")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically (default: deterministic).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    _setup_logging(verbose=bool(args.verbose))

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 2

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (PROJECT_ROOT / "results" / "model_viz" / checkpoint_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", checkpoint_path)
    model = PPO.load(str(checkpoint_path), device="cpu")
    policy_sd = model.policy.state_dict()

    # Weights: stats + plots
    stats = [_param_stats(name, t) for name, t in policy_sd.items()]
    with (out_dir / "param_stats.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(checkpoint_path), "stats": stats}, f, indent=2)

    grouped = _group_params(policy_sd)
    _plot_weight_distributions(
        grouped,
        title=f"Weight distributions: {checkpoint_path.name}",
        out_path=out_dir / "weight_distributions.png",
    )
    _plot_param_norms(
        stats,
        title=f"Parameter L2 norms: {checkpoint_path.name}",
        out_path=out_dir / "param_norms.png",
    )

    # Segment importance (tries to infer using config defaults).
    cfg = Config()
    obs_dim = int(getattr(model.observation_space, "shape", (0,))[0])
    context_dim = int(cfg.analyst.context_dim) if bool(cfg.trading.use_analyst) else 0
    agent_lookback_window = int(cfg.trading.agent_lookback_window)
    analyst_metrics_dim = 5 if bool(cfg.trading.use_analyst) else 0
    n_market_features = obs_dim - (context_dim + 4 + analyst_metrics_dim + 2 + 4 + agent_lookback_window)
    if n_market_features <= 0:
        logger.warning("Skipping input segment importance (failed to infer n_market_features for obs_dim=%d).", obs_dim)
    else:
        segments = _infer_obs_segments(
            obs_dim=obs_dim,
            context_dim=context_dim,
            agent_lookback_window=agent_lookback_window,
            analyst_metrics_dim=analyst_metrics_dim,
            n_market_features=n_market_features,
        )
        _plot_input_segment_importance(
            policy_sd,
            obs_segments=segments,
            out_path=out_dir / "input_segment_importance.png",
            title=f"Input importance by observation segment: {checkpoint_path.name}",
        )

    # Policy rollout
    if not args.no_rollout:
        try:
            env, env_meta = _build_rollout_env(model, cfg=cfg, rollout_steps=int(args.rollout_steps))
            with (out_dir / "env_meta.json").open("w", encoding="utf-8") as f:
                json.dump(env_meta, f, indent=2)

            df = _run_policy_rollout(
                model,
                env,
                steps=int(args.rollout_steps),
                deterministic=not bool(args.stochastic),
            )
            df.to_csv(out_dir / "policy_rollout.csv", index=False)
            _plot_policy_rollout(
                df,
                out_path=out_dir / "policy_rollout.png",
                title=f"Policy rollout: {checkpoint_path.name} (n={len(df):,})",
            )
            _plot_action_frequencies(
                df,
                out_path=out_dir / "action_frequencies.png",
                title=f"Action frequencies: {checkpoint_path.name} (n={len(df):,})",
            )
        except Exception as exc:
            logger.error("Rollout failed (weights plots still generated): %s", exc, exc_info=True)

    logger.info("Wrote outputs to: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

