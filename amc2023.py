
from datasets import load_dataset
import pandas as pd
import random
import numpy as np
from tqdm import tqdm

import sympy as sp


def judge_if_sympy_solveable(answer):
    try:
        int(answer)
        return True
    except:
        pass
    
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

amc23 = load_dataset("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/amc23")["test"]
amc23 = amc23.to_pandas()

amc23_data = []

for index, row in tqdm(amc23.iterrows(), desc="Processing rows"):
    data = process_row(row)
    if data is not None:
        amc23_data.append(data)

print(f"total_size: {len(amc23_data)}")
import json
with open("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/amc23.json", "w") as f:
    json.dump(amc23_data, f, indent=4)
amc23_data = pd.DataFrame(amc23_data)

parquet_path = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/amc23.parquet"
amc23_data.to_parquet(parquet_path)

print(f"first row: {amc23_data.iloc[0]}")
print(f"Length of amc23_data: {len(amc23_data)}")