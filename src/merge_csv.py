"""Merge DATA7901_DR19_MERGED1/2/3.csv into DATA7901_DR19_merged.csv."""
import pandas as pd
from pathlib import Path

tables = Path("input/tables")
parts = [tables / f"DATA7901_DR19_MERGED{i}.csv" for i in range(1, 4)]

dfs = [pd.read_csv(p) for p in parts]
merged = pd.concat(dfs, ignore_index=True)

out = tables / "DATA7901_DR19_merged.csv"
merged.to_csv(out, index=False)
print(f"Merged {sum(len(d) for d in dfs):,} rows → {out} ({len(merged):,} rows, {merged.shape[1]} columns)")
