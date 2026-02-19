"""
Merge proper_setup? and alternate_setup? labels from v1 CSV into v2 CSV.

v1: backscanner_2021_A_plus.csv     (has labels)
v2: backscanner_2021_A_plus_v2.csv  (needs labels carried over)

Key: (date, ticker)
"""

from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

V1_PATH = _DATA_DIR / 'backscanner_2021_A_plus.csv'
V2_PATH = _DATA_DIR / 'backscanner_2021_A_plus_v2.csv'

# 1. Load both CSVs
v1 = pd.read_csv(V1_PATH, dtype=str)
v2 = pd.read_csv(V2_PATH, dtype=str)

print(f"V1 total rows: {len(v1)}")
print(f"V2 total rows: {len(v2)}")
print(f"V1 columns: {list(v1.columns)}")
print(f"V2 columns: {list(v2.columns)}")
print()

# 2. From v1, extract rows with non-empty proper_setup? value
v1_labeled = v1[v1["proper_setup?"].notna() & (v1["proper_setup?"].str.strip() != "")]
print(f"V1 rows with proper_setup? label: {len(v1_labeled)}")
print(f"  - Y: {(v1_labeled['proper_setup?'] == 'Y').sum()}")
print(f"  - N: {(v1_labeled['proper_setup?'] == 'N').sum()}")
print()

# Build lookup: (date, ticker) -> (proper_setup?, alternate_setup?)
label_lookup = {}
for _, row in v1_labeled.iterrows():
    key = (row["date"].strip(), row["ticker"].strip())
    proper = row["proper_setup?"].strip() if pd.notna(row["proper_setup?"]) else ""
    alternate = row["alternate_setup?"].strip() if "alternate_setup?" in row.index and pd.notna(row["alternate_setup?"]) else ""
    label_lookup[key] = (proper, alternate)

print(f"Unique (date, ticker) keys with labels in v1: {len(label_lookup)}")
print()

# 3. For each v2 row, check if (date, ticker) exists in v1 labels
# Ensure v2 has the columns
if "proper_setup?" not in v2.columns:
    v2["proper_setup?"] = ""
if "alternate_setup?" not in v2.columns:
    v2["alternate_setup?"] = ""

carried_over = 0
new_rows = 0
v2_keys = set()

for idx, row in v2.iterrows():
    key = (row["date"].strip(), row["ticker"].strip())
    v2_keys.add(key)
    if key in label_lookup:
        v2.at[idx, "proper_setup?"] = label_lookup[key][0]
        v2.at[idx, "alternate_setup?"] = label_lookup[key][1]
        carried_over += 1
    else:
        new_rows += 1

# 4. Find v1 labeled rows that were dropped (in v1 but not v2)
dropped_keys = set(label_lookup.keys()) - v2_keys
dropped_count = len(dropped_keys)

print("=" * 60)
print("MERGE RESULTS")
print("=" * 60)
print(f"V2 rows that got labels carried over: {carried_over}")
print(f"V2 rows that are NEW (not in v1):     {new_rows}")
print(f"V1 labeled rows DROPPED (not in v2):  {dropped_count}")
print()

if dropped_keys:
    print("Dropped (date, ticker) pairs:")
    for d, t in sorted(dropped_keys):
        proper, alternate = label_lookup[(d, t)]
        print(f"  {d}  {t:10s}  proper={proper}  alternate={alternate}")
    print()

# Show some examples of carried-over labels
print("Sample carried-over labels (first 10):")
labeled_rows = v2[v2["proper_setup?"].notna() & (v2["proper_setup?"].str.strip() != "")]
for _, row in labeled_rows.head(10).iterrows():
    alt = row.get("alternate_setup?", "")
    print(f"  {row['date']}  {row['ticker']:10s}  proper={row['proper_setup?']}  alternate={alt}")

# 5. Write the updated v2 back
# Ensure column order: put proper_setup? and alternate_setup? at the end,
# and keep 'notes' if it exists
cols = [c for c in v2.columns if c not in ("proper_setup?", "alternate_setup?", "notes")]
cols.append("proper_setup?")
cols.append("alternate_setup?")
if "notes" in v2.columns:
    cols.append("notes")

v2 = v2[cols]
v2.to_csv(V2_PATH, index=False)
print(f"\nUpdated v2 written to: {V2_PATH}")
print(f"Final v2 columns: {list(v2.columns)}")
print(f"Final v2 row count: {len(v2)}")
