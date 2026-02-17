#!/usr/bin/env python3
"""
Script to process raw data into normalized features (Parquet) without training models.
"""

import sys
import os
from pathlib import Path
import logging
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device, clear_memory
from scripts.core.run_pipeline import (
    step_1_load_data,
    step_2_resample_timeframes,
    step_3_engineer_features,
    step_3b_normalize_features,
    step_4_train_analyst # Imported but not used
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    config = Config()
    config.paths.ensure_dirs()
    
    logger.info("Starting Data Processing Only...")
    logger.info(f"Raw File: {config.data.raw_file}")
    
    # Step 1: Load data
    df_1m = step_1_load_data(config)

    # Step 2: Resample
    df_5m, df_15m, df_45m = step_2_resample_timeframes(df_1m, config)

    # Step 3: Feature engineering
    df_5m, df_15m, df_45m = step_3_engineer_features(df_5m, df_15m, df_45m, config)
    
    # Define features to normalize
    model_feature_cols = [
            'returns', 'volatility',           # Price dynamics
            'sma_distance',                    # Trend filter
            'dist_to_resistance', 'dist_to_support', # S/R distance
            'sr_strength_r', 'sr_strength_s',  # S/R strength
            'session_asian', 'session_london', 'session_ny',  # Session flags (kept raw)
            'structure_fade',                  # Market structure (continuous)
            'bars_since_bos', 'bars_since_choch',  # Structure recency
            'bos_magnitude', 'choch_magnitude',    # Structure magnitude
            'atr_context',                     # Volatility context (log ATR, kept raw)
    ]
    all_feature_cols = ['open', 'high', 'low', 'close', 'atr', 'chop', 'adx'] + model_feature_cols
    
    # Step 3b: Normalize
    # We pass all_feature_cols so it knows what to keep raw
    step_3b_normalize_features(
        df_5m, df_15m, df_45m, all_feature_cols, config
    )
    
    logger.info("Data processing complete. Parquet files updated in data/processed/")

if __name__ == "__main__":
    main()
