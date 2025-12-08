# eval.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm
from sympy import simplify
from sympy.parsing.latex import parse_latex

# -----------------------------------------------------------
# 1. 单卡 batch 大小，直接在最上面改
# -----------------------------------------------------------
BATCH_SIZE = 4          # 根据显存调整
num_seqs = 32
# -----------------------------------------------------------
# 2. 路径
# -----------------------------------------------------------

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
args = parser.parse_args()

MODEL_PATH = args.model_path
VAL_DATA_PATH = args.dataset_path

# MODEL_PATH = "tmp"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen7grpo8"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen7srpo2r8"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qweb7srpo2r82"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen7grpo16"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen7grpo32"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen1.5math500rflux"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen1.5gsmrflux"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/gsmqwen7grpo8"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/gsmqwen7srpo2r8"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen3srpo2"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/uniform"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/random"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/qwen1.5math"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/Qwen2.5-Math-7B-Instruct"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/math500srpockpt"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/math500grpo16"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/math500grpo8"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/Qwen3-8B"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen3srpo"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen38bgrpo"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen3grpor16"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen3grpor32"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/qwen3grpo2"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/math500srpo2"
# MODEL_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/math500srpo22"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/math500split_val.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2025.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2024.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2023.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktest.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/olymHard.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/olymEasy.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/valAime_data.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/valOlympiads_data.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/amc23.parquet"
# VAL_DATA_PATH = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime.parquet"


# -----------------------------------------------------------
# 3. 工具函数
# -----------------------------------------------------------
def compare_latex_expressions(expr1: str, expr2: str) -> bool:
    # if expr1.strip() == expr2.strip():
    #     return True
    try:
        if int(expr1) == int(expr2):
            return True
    except:
        pass
    try:
        def normalize(expr):
            expr = re.sub(r'\s+', '', expr)
            return expr.replace(' ', '')
        
        if normalize(expr1) == normalize(expr2):
            return True
        
        try:
            sympy1 = parse_latex(expr1)
            sympy2 = parse_latex(expr2)
            return simplify(sympy1 - sympy2) == 0
        except:
            return False
            
    except Exception as e:
        return False

class MathDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        messages = row["prompt"]
        gt = row["reward_model"]["ground_truth"]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, gt

def collate_fn(batch, tokenizer):
    texts, gts = zip(*batch)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    return inputs, gts

# -----------------------------------------------------------
# 4. 主流程
# -----------------------------------------------------------
def main():
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16
    ).to(device)
    torch.manual_seed(42)
    np.random.seed(42)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 数据
    df = pd.read_parquet(VAL_DATA_PATH)
    # df = df.sample(frac=1).reset_index(drop=True)
    df = df.reset_index(drop=True)
    dataset = MathDataset(df, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4
    )

    correct = 0
    incorrect = 0
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'

    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for inputs, gts in pbar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if num_seqs > 1:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.90,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_seqs
                )  # (B * 32, L')
            elif num_seqs == 1:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )  # (B, L')

            # 逐条解码：每个问题有32个回答
            batch_size = len(gts)
            
            for b, gt in enumerate(gts):
                # 获取当前问题的32个回答序列
                start_idx = b * num_seqs
                end_idx = (b + 1) * num_seqs
                question_outputs = outputs[start_idx:end_idx]  # (32, L')
                
                # 检查32个回答中是否有任何一个正确
                found_correct = False
                for seq_idx in range(num_seqs):
                    resp = tokenizer.decode(question_outputs[seq_idx], skip_special_tokens=True)
                    matches = re.findall(pattern, resp)
                    if not matches:
                        continue
                    ans = matches[-1]
                    if compare_latex_expressions(ans, gt):
                        correct += 1
                        found_correct = True
                    else:
                        incorrect += 1
                
                # if found_correct:
                #     correct += 1
                # else:
                #     incorrect += 1

            # 更新进度条
            total = correct + incorrect
            pbar.set_postfix({
                'correct': correct,
                'incorrect': incorrect,
                'accuracy': f"{correct/total*100:.2f}%" if total else "0.00%"
            })

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Total: {total}")
    print(f"Accuracy: {correct/total:.4f} ({correct/total*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU found.")
        exit(1)
    main()