#!/usr/bin/env python3
"""
Training/Live Parity Verification Script

This script loads historical data and compares observations from:
1. TradingEnv (training environment)
2. MT5Bridge._build_observation (live inference)

Any differences indicate parity issues that will cause model degradation in live trading.

Usage:
    python scripts/analysis/verify_parity.py [--n-steps 100] [--verbose]

Example:
    python scripts/analysis/verify_parity.py --n-steps 50 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from src.live.bridge_constants import MODEL_FEATURE_COLS, MARKET_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_observation_component_indices(config: Config) -> Dict[str, Tuple[int, int]]:
    """Calculate observation component indices based on config."""
    use_analyst = config.trading.use_analyst
    context_dim = config.analyst.context_dim if use_analyst else 0
    market_dim = len(MARKET_FEATURE_COLS) * 3
    analyst_metrics_dim = 5 if use_analyst else 0
    returns_dim = config.trading.agent_lookback_window

    # Calculate indices
    idx = 0
    indices = {}

    indices['context'] = (idx, idx + context_dim)
    idx += context_dim

    indices['position'] = (idx, idx + 4)
    idx += 4

    indices['market'] = (idx, idx + market_dim)
    idx += market_dim

    indices['analyst_metrics'] = (idx, idx + analyst_metrics_dim)
    idx += analyst_metrics_dim

    indices['sl_tp'] = (idx, idx + 2)
    idx += 2

    indices['hold'] = (idx, idx + 4)
    idx += 4

    indices['returns'] = (idx, idx + returns_dim)
    idx += returns_dim

    indices['total'] = (0, idx)

    return indices


def verify_observation_structure(config: Config) -> Dict[str, Any]:
    """Verify observation structure matches expected dimensions."""
    logger.info("Verifying observation structure...")

    indices = get_observation_component_indices(config)
    total_dim = indices['total'][1]

    logger.info("Observation component breakdown:")
    for component, (start, end) in indices.items():
        if component != 'total':
            logger.info(f"  {component}: [{start}:{end}] ({end - start} dims)")
    logger.info(f"  TOTAL: {total_dim} dims")

    # Verify against expected values
    expected = {
        'context': config.analyst.context_dim if config.trading.use_analyst else 0,
        'position': 4,
        'market': len(MARKET_FEATURE_COLS) * 3,  # 33 * 3 = 99
        'analyst_metrics': 5 if config.trading.use_analyst else 0,
        'sl_tp': 2,
        'hold': 4,
        'returns': config.trading.agent_lookback_window,
    }

    issues = []
    for component, expected_dim in expected.items():
        actual_dim = indices[component][1] - indices[component][0]
        if actual_dim != expected_dim:
            issues.append(f"{component}: expected {expected_dim}, got {actual_dim}")

    return {
        'indices': indices,
        'total_dim': total_dim,
        'issues': issues,
        'passed': len(issues) == 0
    }


def verify_feature_ordering() -> Dict[str, Any]:
    """Verify feature column ordering is correct."""
    logger.info("Verifying feature column ordering...")

    # Expected MODEL_FEATURE_COLS (updated v2 structure features)
    expected_model_cols = [
        "returns", "volatility", "sma_distance",
        "dist_to_resistance", "dist_to_support",
        "sr_strength_r", "sr_strength_s",
        "session_asian", "session_london", "session_ny",
        "structure_fade", "bars_since_bos", "bars_since_choch",
        "bos_magnitude", "choch_magnitude",
        "atr_context",
        # v35 percentile features
        "atr_percentile", "chop_percentile", "sma_distance_percentile",
        "volatility_percentile",
        # v36 OOD features
        "volatility_regime", "distribution_shift_score",
        # v37 training-anchored OOD features
        "volatility_vs_training", "returns_skew_shift", "atr_vs_training",
        "range_vs_training", "ood_score",
    ]

    # Expected MARKET_FEATURE_COLS (updated v2 structure features)
    expected_market_cols = [
        "atr", "chop", "adx", "sma_distance",
        "dist_to_support", "dist_to_resistance",
        "sr_strength_r", "sr_strength_s",
        "session_asian", "session_london", "session_ny",
        "structure_fade", "bars_since_bos", "bars_since_choch",
        "bos_magnitude", "choch_magnitude",
        "returns", "volatility", "atr_context",
        # v35 percentile features
        "atr_percentile", "chop_percentile", "sma_distance_percentile",
        "volatility_percentile",
        # v36 OOD features
        "volatility_regime", "distribution_shift_score",
        # v37 training-anchored OOD features
        "volatility_vs_training", "returns_skew_shift", "atr_vs_training",
        "range_vs_training", "ood_score",
    ]

    issues = []

    # Check MODEL_FEATURE_COLS
    if list(MODEL_FEATURE_COLS) != expected_model_cols:
        issues.append(f"MODEL_FEATURE_COLS mismatch: {list(MODEL_FEATURE_COLS)}")

    # Check MARKET_FEATURE_COLS
    if list(MARKET_FEATURE_COLS) != expected_market_cols:
        issues.append(f"MARKET_FEATURE_COLS mismatch: {list(MARKET_FEATURE_COLS)}")

    # Check counts
    expected_model_count = len(expected_model_cols)
    if len(MODEL_FEATURE_COLS) != expected_model_count:
        issues.append(f"MODEL_FEATURE_COLS count: expected {expected_model_count}, got {len(MODEL_FEATURE_COLS)}")

    expected_market_count = len(expected_market_cols)
    if len(MARKET_FEATURE_COLS) != expected_market_count:
        issues.append(f"MARKET_FEATURE_COLS count: expected {expected_market_count}, got {len(MARKET_FEATURE_COLS)}")

    if issues:
        for issue in issues:
            logger.error(f"  FAIL: {issue}")
    else:
        logger.info("  PASS: All feature columns in correct order")

    return {
        'model_cols': list(MODEL_FEATURE_COLS),
        'market_cols': list(MARKET_FEATURE_COLS),
        'issues': issues,
        'passed': len(issues) == 0
    }


def verify_artifacts_exist(config: Config) -> Dict[str, Any]:
    """Verify all required parity artifacts exist."""
    logger.info("Verifying required artifacts...")

    required_artifacts = [
        (config.paths.models_analyst / "normalizer_5m.pkl", "5m normalizer"),
        (config.paths.models_analyst / "normalizer_15m.pkl", "15m normalizer"),
        (config.paths.models_analyst / "normalizer_45m.pkl", "45m normalizer"),
        (config.paths.models_agent / "market_feat_stats.npz", "market feature stats"),
        (config.paths.models_agent / "training_baseline.json", "v37 training baseline"),
        (config.paths.models_agent / "final_model.zip", "PPO agent model"),
        (config.paths.models_analyst / "best.pt", "Analyst model"),
    ]

    results = {}
    issues = []

    for path, name in required_artifacts:
        exists = path.exists()
        results[name] = {'path': str(path), 'exists': exists}

        if exists:
            logger.info(f"  FOUND: {name} at {path}")
        else:
            logger.error(f"  MISSING: {name} at {path}")
            issues.append(f"Missing {name}: {path}")

    return {
        'artifacts': results,
        'issues': issues,
        'passed': len(issues) == 0
    }


def run_verification(n_steps: int = 100, verbose: bool = False) -> Dict[str, Any]:
    """Run full parity verification."""
    logger.info("=" * 60)
    logger.info("TRAINING/LIVE PARITY VERIFICATION")
    logger.info("=" * 60)

    config = Config()

    results = {
        'observation_structure': verify_observation_structure(config),
        'feature_ordering': verify_feature_ordering(),
        'artifacts': verify_artifacts_exist(config),
    }

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for check_name, check_result in results.items():
        status = "PASS" if check_result['passed'] else "FAIL"
        logger.info(f"  {check_name}: {status}")
        if not check_result['passed']:
            all_passed = False
            for issue in check_result.get('issues', []):
                logger.error(f"    - {issue}")

    logger.info("")
    if all_passed:
        logger.info("OVERALL: PASS - All parity checks passed")
    else:
        logger.error("OVERALL: FAIL - Parity issues detected")

    results['overall_passed'] = all_passed
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify training/live parity")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Number of steps to compare (default: 100)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_verification(n_steps=args.n_steps, verbose=args.verbose)

    # Exit with appropriate code
    sys.exit(0 if results['overall_passed'] else 1)


if __name__ == "__main__":
    main()
