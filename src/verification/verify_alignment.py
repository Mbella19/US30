import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.data.loader import load_ohlcv
from src.data.components import load_component_data, PRIMARY_COMPONENTS

def verify_alignment():
    print("="*60)
    print("VERIFYING US30 vs COMPONENT DATA ALIGNMENT")
    print("="*60)
    
    config = Config()
    
    # 1. Load Main US30 Data
    us30_path = config.paths.data_raw / config.data.raw_file
    print(f"Loading US30 data from: {us30_path}")

    if not us30_path.exists():
        print(f"ERROR: US30 file not found at {us30_path}")
        return

    try:
        df_us30 = load_ohlcv(us30_path, datetime_format=config.data.datetime_format)
    except Exception as e:
        print(f"ERROR loading US30: {e}")
        return

    print(f"US30 Data: {len(df_us30):,} rows")
    print(f"Range: {df_us30.index.min()} to {df_us30.index.max()}")
    print("-" * 60)
    
    # 2. Load Component Data
    components_dir = config.paths.components_dir
    print(f"Loading Components from: {components_dir}")
    
    if not components_dir.exists():
        print(f"ERROR: Components directory not found at {components_dir}")
        return
        
    components = load_component_data(components_dir)
    
    if not components:
        print("ERROR: No component files loaded.")
        return
        
    print(f"Loaded {len(components)} components: {list(components.keys())}")
    print("-" * 60)
    
    # 3. Analyze Alignment
    print(f"{'COMPONENT':<10} | {'ROWS':<10} | {'START DATE':<20} | {'END DATE':<20} | {'COVERAGE %':<10} | {'MISSING':<10}")
    print("-" * 90)
    
    us30_idx = df_us30.index
    total_us30 = len(us30_idx)
    
    perfect_alignment = True
    
    # Check specific primary components
    for ticker in PRIMARY_COMPONENTS:
        if ticker not in components:
            print(f"{ticker:<10} | {'MISSING':<10} | {'N/A':<20} | {'N/A':<20} | {'0.0%':<10} | {total_us30:<10}")
            perfect_alignment = False
            continue

        df_comp = components[ticker]

        # Intersection
        # We reindex component to match US30 exactly (without filling yet) to see raw overlap
        common = df_comp.reindex(us30_idx)
        valid_count = common['close'].notna().sum()
        missing_count = total_us30 - valid_count
        coverage = (valid_count / total_us30) * 100
        
        start_str = str(df_comp.index.min())[:19]
        end_str = str(df_comp.index.max())[:19]
        
        print(f"{ticker:<10} | {len(df_comp):<10,} | {start_str:<20} | {end_str:<20} | {coverage:6.1f}% | {missing_count:<10,}")
        
        if coverage < 95.0:
            perfect_alignment = False
            
    print("-" * 90)
    
    if perfect_alignment:
        print("\nSUCCESS: High alignment detected (>95%) for all primary components.")
    else:
        print("\nWARNING: Some components have significant gaps or mismatches with US30 data.")
        print("This may affect the accuracy of the Cross-Asset Attention module.")
        print("Recommendation: Ensure component data covers the same historical range as US30.")

if __name__ == "__main__":
    verify_alignment()
