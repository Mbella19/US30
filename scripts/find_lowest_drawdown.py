import pandas as pd
import re
from pathlib import Path

def parse_markdown_table(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the start of the big table "Complete Rankings"
    start_idx = -1
    for i, line in enumerate(lines):
        if "Complete Rankings" in line:
            start_idx = i
            break
    
    if start_idx == -1:
        print("Could not find 'Complete Rankings' section.")
        return

    # Extract table lines
    table_lines = []
    for line in lines[start_idx:]:
        if line.strip().startswith("|"):
            table_lines.append(line)
    
    # Parse into lists
    data = []
    headers = [h.strip() for h in table_lines[0].split('|') if h.strip()]
    
    for line in table_lines[2:]: # Skip header and separator
        row = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(row) == len(headers):
            data.append(row)
            
    df = pd.DataFrame(data, columns=headers)
    
    # Clean up numeric columns
    # 'Steps' might have commas or be 'None'. Coerce errors to NaN and drop.
    df['Steps_Clean'] = pd.to_numeric(df['Steps'].str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Steps_Clean'])
    df['Steps_Clean'] = df['Steps_Clean'].astype(int)
    
    df['Max_DD'] = df['Max DD%'].astype(float)
    df['Return_Pct'] = df['Return%'].astype(float)
    df['Net_PnL'] = df['Net PnL'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Sort by Max DD ascending
    df_sorted = df.sort_values('Max_DD', ascending=True)
    
    print("\nTop 5 Lowest Drawdown Checkpoints:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Steps':<15} {'Max DD%':<10} {'Return%':<10} {'Net PnL':<15}")
    print("-" * 60)
    
    for i, row in df_sorted.head(5).iterrows():
        print(f"{i+1:<5} {row['Steps']:<15} {row['Max_DD']:<10.2f} {row['Return_Pct']:<10.2f} {row['Net PnL']:<15}")

if __name__ == "__main__":
    # Use path relative to project root
    project_root = Path(__file__).parent.parent
    parse_markdown_table(project_root / "results" / "checkpoint_performance_ranked.md")
