"""
00_inspect_columns.py
Inspect column names and basic stats from all 3 processed Excel files.
Run this first to verify column mappings before building indices.
"""
import pandas as pd
import os

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ModeloDatos", "ModeloDatos", "data", "processed"
)

files = {
    "ECV": "ECV_Procesado.xlsx",
    "IML": "IML_Procesado.xlsx",
    "IPM": "IPM_Procesado.xlsx",
}

for name, fname in files.items():
    path = os.path.join(DATA_PATH, fname)
    df = pd.read_excel(path)
    print(f"\n{'='*60}")
    print(f"  {name}  —  {fname}")
    print(f"{'='*60}")
    print(f"  Shape : {df.shape}")
    print(f"  Cols  : {df.columns.tolist()}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Sample (first 3 rows):\n{df.head(3).to_string()}")
    print()
