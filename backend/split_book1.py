import pandas as pd
from pathlib import Path

# Paths
base = Path(__file__).parent
input_file = base / "Book1.xlsx"
output_file = base / "Book1_split.xlsx"

# Read Excel
df = pd.read_excel(input_file)
# Get the single column name
col = df.columns[0]
# Split into two columns by first '-'
split_df = df[col].astype(str).str.split('-', n=1, expand=True)
split_df.columns = ['col1', 'col2']
# Save result, overwrite existing
split_df.to_excel(output_file, index=False)
print(f"Split written to {output_file}")
