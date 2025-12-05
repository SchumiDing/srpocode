
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
    question = row_dict["problem"]
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

aime2023 = load_dataset("data/aime2023")["train"]
aime2023 = aime2023.to_pandas()

aime2023_data = []

for index, row in tqdm(aime2023.iterrows(), desc="Processing rows"):
    data = process_row(row)
    if data is not None:
        aime2023_data.append(data)

print(f"total_size: {len(aime2023_data)}")
import json
with open("data/aime2023.json", "w") as f:
    json.dump(aime2023_data, f, indent=4)
aime2023_data = pd.DataFrame(aime2023_data)

parquet_path = "data/aime2023.parquet"
aime2023_data.to_parquet(parquet_path)

print(f"first row: {aime2023_data.iloc[0]}")
print(f"Length of aime2023_data: {len(aime2023_data)}")