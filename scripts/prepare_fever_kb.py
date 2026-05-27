import pandas as pd
import os

df = pd.read_csv(
    "data/raw/fever/fever_data.csv"
)

out_dir="data/raw/txt"

os.makedirs(
    out_dir,
    exist_ok=True
)

for i,row in df.iterrows():

    evidence=row["evidence"]

    if pd.isna(evidence):
        continue

    with open(
        f"{out_dir}/fever_{i}.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(str(evidence))

print("Knowledge files created")