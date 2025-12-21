import argparse
import json
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sympy import simplify
from sympy.parsing.latex import parse_latex
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -----------------------------------------------------------
# 1. 单卡 batch 大小（提示并行推理批量大小）
# -----------------------------------------------------------
BATCH_SIZE_DEFAULT = 32       # 提升吞吐，显存够可调大
TP_SIZE_DEFAULT = 4           # 张量并行卡数
MAX_MODEL_LEN_DEFAULT = 4096
MAX_NEW_TOKENS_DEFAULT = 2048

# -----------------------------------------------------------
# 2. 参数
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument(
    "--test_mode",
    type=str,
    default="all",
    choices=["all", "pass@1", "pass@8", "pass@16", "pass@32", "pass@64", "pass@128", "pass@256", "mean32"],
    help="Test mode: all, pass@1, pass@8, pass@16, pass@32, pass@64, pass@128, pass@256, or mean32",
)
parser.add_argument("--output_json", type=str, default=None, help="Output results to JSON file (for batch processing)")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="vLLM 侧一次送入的 prompt 数")
parser.add_argument("--tp_size", type=int, default=TP_SIZE_DEFAULT, help="tensor parallel size，多卡并行")
parser.add_argument("--max_model_len", type=int, default=MAX_MODEL_LEN_DEFAULT, help="上下文最大长度")
parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT, help="生成最大新 token 数")
args, unknown_args = parser.parse_known_args()
if unknown_args:
    print(f"警告：检测到未知参数 {unknown_args} ，将被忽略。")

MODEL_PATH = args.model_path
VAL_DATA_PATH = args.dataset_path
TEST_MODE = args.test_mode
OUTPUT_JSON = args.output_json
BATCH_SIZE = args.batch_size
TP_SIZE = args.tp_size
MAX_MODEL_LEN = args.max_model_len
MAX_NEW_TOKENS = args.max_new_tokens

# -----------------------------------------------------------
# 3. 工具函数
# -----------------------------------------------------------
def compare_latex_expressions(expr1: str, expr2: str) -> bool:
    try:
        if int(expr1) == int(expr2):
            return True
    except Exception:
        pass
    try:
        def normalize(expr):
            expr = re.sub(r"\s+", "", expr)
            return expr.replace(" ", "")

        if normalize(expr1) == normalize(expr2):
            return True

        try:
            sympy1 = parse_latex(expr1)
            sympy2 = parse_latex(expr2)
            return simplify(sympy1 - sympy2) == 0
        except Exception:
            return False

    except Exception:
        return False


def chunk_list(xs: List, size: int):
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


# -----------------------------------------------------------
# 4. 主流程（vLLM 张量并行）
# -----------------------------------------------------------
def build_sampling(num_seqs: int, is_mean: bool) -> SamplingParams:
    # pass@1: 不采样；其它：采样
    if num_seqs == 1:
        return SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
            seed=42,
        )
    return SamplingParams(
        n=num_seqs,
        temperature=0.7,
        top_p=0.90,
        max_tokens=MAX_NEW_TOKENS,
        seed=42,
    )


def evaluate_mode(
    llm: LLM,
    tokenizer,
    prompts: List[str],
    gts: List[str],
    num_seqs: int,
    mode_name: str,
    is_mean: bool,
) -> Tuple[int, int, int, float, str]:
    correct = 0
    incorrect = 0
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    sampling_params = build_sampling(num_seqs, is_mean)

    pbar = tqdm(total=len(prompts), desc=f"Evaluating {mode_name}", unit="batch")
    for batch_prompts, batch_gts in zip(chunk_list(prompts, BATCH_SIZE), chunk_list(gts, BATCH_SIZE)):
        results = llm.generate(batch_prompts, sampling_params=sampling_params)
        # results 与 batch_prompts 对齐
        for res, gt in zip(results, batch_gts):
            generations = res.outputs  # list of Generation
            if num_seqs == 1:
                resp = generations[0].text
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
                if is_mean:
                    correct_count = 0
                    for gen in generations:
                        resp = gen.text
                        matches = re.findall(pattern, resp)
                        if matches:
                            ans = matches[-1]
                            if compare_latex_expressions(ans, gt):
                                correct_count += 1
                    correct += correct_count
                    incorrect += (num_seqs - correct_count)
                else:
                    found = False
                    for gen in generations:
                        resp = gen.text
                        matches = re.findall(pattern, resp)
                        if matches:
                            ans = matches[-1]
                            if compare_latex_expressions(ans, gt):
                                found = True
                                break
                    if found:
                        correct += 1
                    else:
                        incorrect += 1

        pbar.update(len(batch_prompts))
        total = correct + incorrect
        accuracy = correct / total if total > 0 else 0.0
        pbar.set_postfix(
            correct=correct,
            incorrect=incorrect,
            accuracy=f"{accuracy*100:.2f}%",
        )
    pbar.close()

    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0.0
    description = f"Mean accuracy over {num_seqs} samples" if is_mean else f"Pass@{num_seqs}"
    return correct, incorrect, total, accuracy, description


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    available_gpus = torch.cuda.device_count()
    if available_gpus < TP_SIZE:
        raise RuntimeError(f"请求 TP_SIZE={TP_SIZE}，但仅检测到 {available_gpus} 张 GPU，请检查资源或 CUDA_VISIBLE_DEVICES。")
    print(f"使用 vLLM 张量并行：TP={TP_SIZE}，模型={MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=False,
        tensor_parallel_size=TP_SIZE,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    # 数据
    df = pd.read_parquet(VAL_DATA_PATH).reset_index(drop=True)
    messages_list = df["prompt"].tolist()
    gts = df["reward_model"].apply(lambda x: x["ground_truth"]).tolist()
    prompts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]

    # 测试模式
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
        print("\n" + "=" * 60)
        print(f"Running {mode_name} evaluation")
        print("=" * 60)

        correct, incorrect, total, accuracy, description = evaluate_mode(
            llm, tokenizer, prompts, gts, num_seqs, mode_name, is_mean
        )

        results[mode_name] = {
            "correct": correct,
            "incorrect": incorrect,
            "total": total,
            "accuracy": accuracy,
            "description": description,
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

    # 汇总
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    for mode_name in ["pass@1", "pass@8", "pass@16", "pass@32", "pass@64", "pass@128", "pass@256", "mean32"]:
        if mode_name in results:
            r = results[mode_name]
            if mode_name == "mean32":
                print(f"{mode_name:12s}: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%) - Mean accuracy")
            else:
                print(f"{mode_name:12s}: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%) - Pass@{mode_name.split('@')[1]}")
    print("=" * 60)

    # 输出 JSON
    if OUTPUT_JSON:
        output_data = {
            "model_path": MODEL_PATH,
            "dataset_path": VAL_DATA_PATH,
            "results": {k: {"accuracy": float(v["accuracy"])} for k, v in results.items()},
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n结果已保存到 JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU found.")
        exit(1)
    main()
