
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
            "data_source": "olymmath",
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

olymHard = load_dataset("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/OlymMath", 'qwq-32b')["en_hard"]
olymHard = olymHard.to_pandas()

olymHard_data = []

for index, row in tqdm(olymHard.iterrows(), desc="Processing rows"):
    if row["response_id"]!=0:
        continue
    data = process_row(row)
    if data is not None:
        olymHard_data.append(data)
        
import json
with open("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/olymHard.json", "w") as f:
    json.dump(olymHard_data, f, indent=4)
olymHard_data = pd.DataFrame(olymHard_data)

parquet_path = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/olymHard.parquet"
olymHard_data.to_parquet(parquet_path)

print(f"first row: {olymHard_data.iloc[0]}")
print(f"Length of olymHard_data: {len(olymHard_data)}")


olymEasy = load_dataset("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/OlymMath", 'qwq-32b')["en_easy"]
olymEasy = olymEasy.to_pandas()

olymEasy_data = []

for index, row in tqdm(olymEasy.iterrows(), desc="Processing rows"):
    if row["response_id"]!=0:
        continue
    data = process_row(row)
    if data is not None:
        olymEasy_data.append(data)
        
with open("/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/olymEasy.json", "w") as f:
    json.dump(olymEasy_data, f, indent=4)
olymEasy_data = pd.DataFrame(olymEasy_data)

parquet_path = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/olymEasy.parquet"
olymEasy_data.to_parquet(parquet_path)

print(f"first row: {olymEasy_data.iloc[0]}")
print(f"Length of olymEasy_data: {len(olymEasy_data)}")
