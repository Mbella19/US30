#!/usr/bin/env python3
"""
Run the MT5 bridge with visualization server for live trading frontend.

This starts:
1. WebSocket visualization server (default port 8765)
2. MT5 TCP bridge server (default port 5555)

The visualization server streams real neural network activations
to the React frontend for real-time visualization.

Usage:
    python scripts/core/run_visualizer.py
    python scripts/core/run_visualizer.py --dry-run
    python scripts/core/run_visualizer.py --viz-port 8766 --port 5555
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path
from queue import Queue

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_visualization_server(viz_queue: Queue, host: str, port: int) -> None:
    """
    Run the visualization WebSocket server in its own thread.

    Args:
        viz_queue: Queue to receive visualization data from bridge
        host: Host address for WebSocket server
        port: Port for WebSocket server
    """
    from src.live.visualization_server import VisualizationServer

    server = VisualizationServer(host=host, port=port)

    # Share the queue with the server
    server.data_queue = viz_queue

    print(f"Starting visualization server on ws://{host}:{port}")

    # Run the async server
    asyncio.run(server.run())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MT5 bridge with visualization server for live trading frontend"
    )

    # MT5 Bridge options
    parser.add_argument("--host", default="127.0.0.1", help="MT5 bridge host")
    parser.add_argument("--port", type=int, default=5555, help="MT5 bridge port")
    parser.add_argument("--symbol", default="US30", help="Trading symbol")
    parser.add_argument("--lot-scale", type=float, default=1.0, help="Scale model size to MT5 lots")
    parser.add_argument("--min-m1-rows", type=int, default=60 * 24 * 30, help="Warmup minutes")
    parser.add_argument("--min-history-days", type=int, default=30, help="Warmup days")
    parser.add_argument("--history-dir", default="data/live", help="Live bars directory")
    parser.add_argument("--dry-run", action="store_true", help="Run inference without trading")
    parser.add_argument("--log-dir", default=None, help="Log directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model", default=None, help="Custom model checkpoint path")

    # Visualization server options
    parser.add_argument("--viz-host", default="localhost", help="Visualization server host")
    parser.add_argument("--viz-port", type=int, default=8765, help="Visualization server port")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization server")

    args = parser.parse_args()

    # Configure logging
    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Import after argparse for fast --help
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

    viz_queue = None
    viz_thread = None

    if not args.no_viz:
        # Create shared queue for visualization data
        viz_queue = Queue()

        # Start visualization server in background thread
        viz_thread = threading.Thread(
            target=run_visualization_server,
            args=(viz_queue, args.viz_host, args.viz_port),
            daemon=True,
        )
        viz_thread.start()

        print(f"\n{'='*60}")
        print("LIVE TRADING VISUALIZATION")
        print(f"{'='*60}")
        print(f"Visualization server: ws://{args.viz_host}:{args.viz_port}")
        print(f"MT5 bridge server:    tcp://{args.host}:{args.port}")
        print(f"Frontend URL:         http://localhost:5173")
        print(f"{'='*60}")
        print("\nTo start the frontend:")
        print("  cd frontend && npm install && npm run dev")
        print(f"{'='*60}\n")

    # Run the MT5 bridge (blocking)
    run_mt5_bridge(
        bridge_cfg,
        system_cfg=system_cfg,
        log_dir=args.log_dir,
        viz_queue=viz_queue,
    )


if __name__ == "__main__":
    main()
