"""
Check for sequences of identical close prices indicating stale or corrupt data.

Checks: Close price sequences in processed data (flags long runs of identical values)
Related: verify_gap_removal.py (checks for time discontinuities in raw data)

Usage:
    python scripts/analysis/check_flat_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config.settings import Config

config = Config()
path = config.paths.data_processed / 'features_15m.parquet'

print(f"Loading {path}...")
df = pd.read_parquet(path)

# Check the last 30 days (approx 2880 bars)
tail = df.tail(2880)
closes = tail['close'].values

# Find sequences of identical values
diffs = np.diff(closes)
zeros = (diffs == 0).astype(int)

# Find longest run of zeros
max_run = 0
current_run = 0
for z in zeros:
    if z == 1:
        current_run += 1
    else:
        max_run = max(max_run, current_run)
        current_run = 0
max_run = max(max_run, current_run)

print(f"Total bars checked: {len(closes)}")
print(f"Longest sequence of identical prices: {max_run} bars")
print(f"Number of zero-diff steps: {np.sum(zeros)}")

# Print a sample of the flat region if it exists
if max_run > 10:
    print("\nSample of flat region:")
    # Find index of long run
    # This is a quick hack, might find the first one
    for i in range(len(zeros) - 10):
        if np.all(zeros[i:i+10] == 1):
            print(tail.iloc[i:i+15][['timestamp', 'close']])
            break
