from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

print("Downloading FEVER dataset...")

dataset = load_dataset("fever", "v1.0")

print(dataset)

train = dataset["train"]

records=[]

limit=2000

for item in tqdm(train.select(range(min(limit, len(train))))):

    claim=item.get("claim","")
    label=item.get("label","")

    evidence_text=[]

    evidence=item.get("evidence",[])

    for ev_group in evidence:
        for ev in ev_group:

            if len(ev)>=4:

                page=str(ev[2])
                sentence_id=str(ev[3])

                evidence_text.append(
                    f"{page} sentence {sentence_id}"
                )

    records.append({

        "claim":claim,
        "label":label,
        "evidence":" ".join(evidence_text)

    })


os.makedirs(
    "data/raw/fever",
    exist_ok=True
)

df=pd.DataFrame(records)

output="data/raw/fever/fever_data.csv"

df.to_csv(
    output,
    index=False
)

print(f"\nSaved: {output}")
print(df.head())