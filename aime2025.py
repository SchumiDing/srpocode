
from datasets import load_dataset
import pandas as pd
import random
import numpy as np
from tqdm import tqdm

import sympy as sp


def judge_if_sympy_solveable(answer):
    try:
        sp.sympify(answer)
        return True
    except:
        return False

def process_row(row_dict):
    question = row_dict["question"]
    ground_truth = row_dict["answer"]
    instruction = 'Let\'s think step by step and put the final answer in the \\boxed{} tag. Do not repeat any sentences in the answer, and keep only one \\boxed{} tag which contains the final answer.'
    if judge_if_sympy_solveable(ground_truth):
        return {
            "data_source": "deepmath",
            "prompt": [
                {
                    "role": "user",
                    "content": question + "\n" + instruction
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            }
        }
    else:
        return None

aime2025I = load_dataset("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/aime2025", "AIME2025-I")["test"]
aime2025II = load_dataset("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/aime2025", "AIME2025-II")["test"]
aime2025I = aime2025I.to_pandas()
aime2025II = aime2025II.to_pandas()
aime2025 = pd.concat([aime2025I, aime2025II])

aime2025_data = []

for index, row in tqdm(aime2025.iterrows(), desc="Processing rows"):
    data = process_row(row)
    if data is not None:
        aime2025_data.append(data)

print(f"total_size: {len(aime2025_data)}")
import json
with open("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2025.json", "w") as f:
    json.dump(aime2025_data, f, indent=4)
aime2025_data = pd.DataFrame(aime2025_data)

parquet_path = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2025.parquet"
aime2025_data.to_parquet(parquet_path)

print(f"first row: {aime2025_data.iloc[0]}")