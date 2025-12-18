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
import json

# -----------------------------------------------------------
# 1. 单卡 batch 大小，直接在最上面改
# -----------------------------------------------------------
BATCH_SIZE = 4          # 根据显存调整
# -----------------------------------------------------------
# 2. 路径
# -----------------------------------------------------------

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--test_mode", type=str, default="all", 
                    choices=["all", "pass@1", "pass@8", "pass@16", "pass@32", "pass@64", "pass@128", "pass@256", "mean32"],
                    help="Test mode: all, pass@1, pass@8, pass@16, pass@32, pass@64, pass@128, pass@256, or mean32")
parser.add_argument("--output_json", type=str, default=None,
                    help="Output results to JSON file (for batch processing)")
args = parser.parse_args()

MODEL_PATH = args.model_path
VAL_DATA_PATH = args.dataset_path
TEST_MODE = args.test_mode
OUTPUT_JSON = args.output_json

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
def evaluate_mode(model, tokenizer, loader, num_seqs, mode_name, is_mean=False):
    """评估单个模式"""
    device = torch.device("cuda:0")
    correct = 0
    incorrect = 0
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'

    pbar = tqdm(loader, desc=f"Evaluating {mode_name}", unit="batch")
    with torch.no_grad():
        for inputs, gts in pbar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if num_seqs == 1:
                # pass@1: 贪心解码
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )  # (B, L')
            else:
                # pass@k 或 mean32: 采样生成
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.90,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_seqs
                )  # (B * num_seqs, L')

            batch_size = len(gts)
            
            for b, gt in enumerate(gts):
                if num_seqs == 1:
                    # pass@1: 单个回答
                    resp = tokenizer.decode(outputs[b], skip_special_tokens=True)
                    matches = re.findall(pattern, resp)
                    if matches:
                        ans = matches[-1]
                        if compare_latex_expressions(ans, gt):
                            correct += 1
                        else:
                            incorrect += 1
                    else:
                        incorrect += 1
                else:
                    # pass@k 或 mean32: 多个回答
                    start_idx = b * num_seqs
                    end_idx = (b + 1) * num_seqs
                    question_outputs = outputs[start_idx:end_idx]  # (num_seqs, L')
                    
                    if is_mean:
                        # mean32: 计算所有回答的平均正确率
                        correct_count = 0
                        for seq_idx in range(num_seqs):
                            resp = tokenizer.decode(question_outputs[seq_idx], skip_special_tokens=True)
                            matches = re.findall(pattern, resp)
                            if matches:
                                ans = matches[-1]
                                if compare_latex_expressions(ans, gt):
                                    correct_count += 1
                        # mean32 模式下，每个问题贡献 correct_count/num_seqs 的正确率
                        correct += correct_count
                        incorrect += (num_seqs - correct_count)
                    else:
                        # pass@k: 检查是否有任何一个正确
                        found_correct = False
                        for seq_idx in range(num_seqs):
                            resp = tokenizer.decode(question_outputs[seq_idx], skip_special_tokens=True)
                            matches = re.findall(pattern, resp)
                            if matches:
                                ans = matches[-1]
                                if compare_latex_expressions(ans, gt):
                                    found_correct = True
                                    break
                        
                        if found_correct:
                            correct += 1
                        else:
                            incorrect += 1

            # 更新进度条
            total = correct + incorrect
            if is_mean:
                # mean32: accuracy = correct / total, 其中 total = num_questions * num_seqs
                accuracy = correct / total if total > 0 else 0.0
            else:
                # pass@k: accuracy = correct / num_questions
                accuracy = correct / total if total > 0 else 0.0
            pbar.set_postfix({
                'correct': correct,
                'incorrect': incorrect,
                'accuracy': f"{accuracy*100:.2f}%"
            })

    total = correct + incorrect
    if is_mean:
        # mean32: accuracy = correct / total, 其中 total = num_questions * num_seqs
        accuracy = correct / total if total > 0 else 0.0
        return correct, incorrect, total, accuracy, f"Mean accuracy over {num_seqs} samples"
    else:
        # pass@k: accuracy = correct / num_questions
        accuracy = correct / total if total > 0 else 0.0
        return correct, incorrect, total, accuracy, f"Pass@{num_seqs}"

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
    df = df.reset_index(drop=True)
    dataset = MathDataset(df, tokenizer)
    
    # 定义所有测试模式
    test_modes = []
    if TEST_MODE == "all":
        test_modes = [
            ("pass@1", 1, False),
            ("pass@8", 8, False),
            ("pass@16", 16, False),
            ("pass@32", 32, False),
            ("pass@64", 64, False),
            ("pass@128", 128, False),
            ("pass@256", 256, False),
            ("mean32", 32, True),
        ]
    elif TEST_MODE == "pass@1":
        test_modes = [("pass@1", 1, False)]
    elif TEST_MODE == "pass@8":
        test_modes = [("pass@8", 8, False)]
    elif TEST_MODE == "pass@16":
        test_modes = [("pass@16", 16, False)]
    elif TEST_MODE == "pass@32":
        test_modes = [("pass@32", 32, False)]
    elif TEST_MODE == "pass@64":
        test_modes = [("pass@64", 64, False)]
    elif TEST_MODE == "pass@128":
        test_modes = [("pass@128", 128, False)]
    elif TEST_MODE == "pass@256":
        test_modes = [("pass@256", 256, False)]
    elif TEST_MODE == "mean32":
        test_modes = [("mean32", 32, True)]

    results = {}
    
    for mode_name, num_seqs, is_mean in test_modes:
        print(f"\n{'='*60}")
        print(f"Running {mode_name} evaluation")
        print(f"{'='*60}")
        
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
            num_workers=4
        )
        
        correct, incorrect, total, accuracy, description = evaluate_mode(
            model, tokenizer, loader, num_seqs, mode_name, is_mean
        )
        
        results[mode_name] = {
            'correct': correct,
            'incorrect': incorrect,
            'total': total,
            'accuracy': accuracy,
            'description': description
        }
        
        print(f"\n{mode_name} Results:")
        print(f"  {description}")
        print(f"  Correct: {correct}")
        print(f"  Incorrect: {incorrect}")
        print(f"  Total: {total}")
        if is_mean:
            print(f"  Mean Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print(f"  Pass@{num_seqs}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 打印汇总结果
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS SUMMARY")
    print("="*60)
    for mode_name in ["pass@1", "pass@8", "pass@16", "pass@32", "pass@64", "pass@128", "pass@256", "mean32"]:
        if mode_name in results:
            r = results[mode_name]
            if mode_name == "mean32":
                print(f"{mode_name:12s}: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%) - Mean accuracy")
            else:
                print(f"{mode_name:12s}: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%) - Pass@{mode_name.split('@')[1]}")
    print("="*60)
    
    # 如果指定了 JSON 输出，保存结果
    if OUTPUT_JSON:
        output_data = {
            'model_path': MODEL_PATH,
            'dataset_path': VAL_DATA_PATH,
            'results': {k: {'accuracy': float(v['accuracy'])} for k, v in results.items()}
        }
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n结果已保存到 JSON: {OUTPUT_JSON}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU found.")
        exit(1)
    main()