#!/usr/bin/env python3
"""
Run the MT5 ↔ Python live bridge server.

This starts a TCP server that accepts a length-prefixed JSON payload from an MT5 EA
and responds with length-prefixed JSON containing {action,size,sl,tp}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path (so `config/` and `src/` are importable)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MT5 ↔ Python bridge server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--symbol", default="US30", help="Main symbol name (must match EA)")
    parser.add_argument("--lot-scale", type=float, default=1.0, help="Scale model size to MT5 lots")
    parser.add_argument("--min-m1-rows", type=int, default=60 * 24 * 30, help="Warmup minutes before trading")
    parser.add_argument("--min-history-days", type=int, default=30, help="Warmup calendar days before trading")
    parser.add_argument("--history-dir", default="data/live", help="Where to persist live bars")
    parser.add_argument("--dry-run", action="store_true", help="Run inference but never trade (noop responses)")
    parser.add_argument("--log-dir", default=None, help="Optional log directory")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging for diagnostics")
    parser.add_argument("--model", default=None, help="Path to custom model checkpoint (e.g., models/agent/checkpoints/model_1000000_steps.zip)")
    args = parser.parse_args()

    # Configure logging level based on --debug flag
    import logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger("src.live.mt5_bridge").setLevel(logging.DEBUG)

    # Delay heavy imports until AFTER argparse (keeps `--help` fast).
    from config.settings import Config
    from src.live.mt5_bridge import BridgeConfig, run_mt5_bridge

    system_cfg = Config()

    bridge_cfg = BridgeConfig(
        host=args.host,
        port=args.port,
        main_symbol=args.symbol,
        lot_scale=args.lot_scale,
        min_m1_rows=args.min_m1_rows,
        min_history_days=args.min_history_days,
        history_dir=Path(args.history_dir),
        dry_run=args.dry_run,
        model_path=Path(args.model) if args.model else None,
    )

    run_mt5_bridge(bridge_cfg, system_cfg=system_cfg, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
