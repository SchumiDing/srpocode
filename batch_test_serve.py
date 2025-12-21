#!/usr/bin/env python3
"""
批量测试脚本（使用 vLLM serve）：对所有模型和数据集组合进行测试
每个模型只启动一次服务（8个GPU，8个数据并行），然后测试所有数据集和评估方法
"""
import os
import json
import subprocess
import time
import csv
import re
import signal
import requests
from typing import List, Tuple
import pandas as pd
from sympy import simplify
from sympy.parsing.latex import parse_latex
from tqdm import tqdm
from transformers import AutoTokenizer

# 模型列表
MODELS = [
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen7grpo",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen7srpo2",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen7rflux",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen7srpo3",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen1.5grpo",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen1.5srpo2",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen1.5rflux",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/trained_models/qwen1.5srpo3",
]

# 数据集列表
DATASETS = [
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2023.parquet",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2024.parquet",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2025.parquet",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/amc23.parquet",
    "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/math500split_val.parquet",
]

# 输出文件夹路径
OUTPUT_DIR = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_evaluation_results_serve"

# 测试指标列表
METRICS = ["pass@1", "pass@8", "pass@16", "pass@32", "pass@64", "pass@128", "pass@256", "mean32"]

# vLLM 配置
VLLM_DATA_PARALLEL_SIZE = 8  # 数据并行大小（8个GPU）
VLLM_BATCH_SIZE = 128         # vLLM 侧批量大小，可按显存调大
VLLM_MAX_NEW_TOKENS = 2048    # 控制生成长度，减少显存与耗时
VLLM_MAX_MODEL_LEN = 4096     # 上下文最大长度
VLLM_PORT = 8000              # vLLM 服务端口
VLLM_HOST = "localhost"       # vLLM 服务地址

# 服务启动等待时间（秒）
SERVICE_STARTUP_WAIT = 120


def extract_model_name(model_path):
    """从模型路径中提取模型名称"""
    return os.path.basename(model_path.rstrip('/'))


def extract_dataset_name(dataset_path):
    """从数据集路径中提取数据集名称"""
    filename = os.path.basename(dataset_path)
    return os.path.splitext(filename)[0]


def compare_latex_expressions(expr1: str, expr2: str) -> bool:
    """比较两个 LaTeX 表达式是否相等"""
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
    """将列表分块"""
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def wait_for_service(url: str, max_wait: int = 300):
    """等待服务启动"""
    print(f"等待服务启动: {url}")
    for i in range(max_wait):
        try:
            # 尝试访问健康检查端点或模型列表端点
            response = requests.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"✓ 服务已启动")
                return True
        except Exception:
            pass
        if i % 10 == 0:
            print(f"等待中... ({i}/{max_wait}秒)")
        time.sleep(1)
    return False


def call_vllm_api(prompts: List[str], num_seqs: int, is_mean: bool, api_url: str):
    """通过 API 调用 vLLM 服务（使用 OpenAI 兼容 API）"""
    # 构建请求参数
    if num_seqs == 1:
        temperature = 0.0
        top_p = 1.0
        n = 1
    else:
        temperature = 0.7
        top_p = 0.90
        n = num_seqs
    
    # 使用 OpenAI 兼容的 API 格式
    # 批量处理所有 prompts
    all_outputs = []
    
    for prompt in prompts:
        request_data = {
            "model": "default",
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": VLLM_MAX_NEW_TOKENS,
            "stop": None,
        }
        
        try:
            response = requests.post(
                f"{api_url}/v1/completions",
                json=request_data,
                timeout=600  # 10分钟超时
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析结果
            prompt_outputs = []
            if "choices" in result:
                for choice in result["choices"]:
                    text = choice.get("text", "")
                    prompt_outputs.append(text)
            else:
                # 如果没有结果，填充空字符串
                prompt_outputs = [""] * n
            
            # 确保返回正确数量的输出
            while len(prompt_outputs) < n:
                prompt_outputs.append("")
            
            all_outputs.append(prompt_outputs[:n])
        except Exception as e:
            print(f"API 调用错误: {e}")
            # 返回空结果
            all_outputs.append([""] * n)
    
    return all_outputs


def evaluate_mode(
    tokenizer,
    prompts: List[str],
    gts: List[str],
    num_seqs: int,
    mode_name: str,
    is_mean: bool,
    api_url: str,
) -> Tuple[int, int, int, float, str]:
    """评估单个模式"""
    correct = 0
    incorrect = 0
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"

    pbar = tqdm(total=len(prompts), desc=f"Evaluating {mode_name}", unit="batch")
    for batch_prompts, batch_gts in zip(chunk_list(prompts, VLLM_BATCH_SIZE), chunk_list(gts, VLLM_BATCH_SIZE)):
        try:
            # 调用 API
            batch_outputs = call_vllm_api(batch_prompts, num_seqs, is_mean, api_url)
            
            # 处理结果
            for outputs, gt in zip(batch_outputs, batch_gts):
                if num_seqs == 1:
                    resp = outputs[0] if outputs else ""
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
                        for resp in outputs:
                            matches = re.findall(pattern, resp)
                            if matches:
                                ans = matches[-1]
                                if compare_latex_expressions(ans, gt):
                                    correct_count += 1
                        correct += correct_count
                        incorrect += (num_seqs - correct_count)
                    else:
                        found = False
                        for resp in outputs:
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
        except Exception as e:
            print(f"评估错误: {e}")
            incorrect += len(batch_prompts)

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


def start_vllm_service(model_path: str) -> subprocess.Popen:
    """启动 vLLM 服务"""
    print(f"\n{'='*80}")
    print(f"启动 vLLM 服务: {extract_model_name(model_path)}")
    print(f"{'='*80}")
    
    cmd = [
        "vllm", "serve", model_path,
        "--data-parallel-size", str(VLLM_DATA_PARALLEL_SIZE),
        "--max-model-len", str(VLLM_MAX_MODEL_LEN),
        "--port", str(VLLM_PORT),
        "--host", VLLM_HOST,
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 启动服务进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # 创建新的进程组
    )
    
    return process


def stop_vllm_service(process: subprocess.Popen):
    """停止 vLLM 服务"""
    print(f"\n停止 vLLM 服务...")
    try:
        # 终止整个进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
        print("✓ 服务已停止")
    except subprocess.TimeoutExpired:
        print("强制终止服务...")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
    except Exception as e:
        print(f"停止服务时出错: {e}")


def test_model_on_all_datasets(model_path: str, api_url: str):
    """在单个模型上测试所有数据集和评估方法"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试模式
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
    
    # 存储所有结果
    all_results = {}
    
    # 对每个数据集进行测试
    for dataset_path in DATASETS:
        dataset_name = extract_dataset_name(dataset_path)
        print(f"\n{'='*80}")
        print(f"测试数据集: {dataset_name}")
        print(f"{'='*80}")
        
        # 加载数据
        df = pd.read_parquet(dataset_path).reset_index(drop=True)
        messages_list = df["prompt"].tolist()
        gts = df["reward_model"].apply(lambda x: x["ground_truth"]).tolist()
        prompts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        dataset_results = {}
        
        # 对每个评估模式进行测试
        for mode_name, num_seqs, is_mean in test_modes:
            print(f"\n{'='*60}")
            print(f"Running {mode_name} evaluation on {dataset_name}")
            print(f"{'='*60}")
            
            try:
                correct, incorrect, total, accuracy, description = evaluate_mode(
                    tokenizer, prompts, gts, num_seqs, mode_name, is_mean, api_url
                )
                
                dataset_results[mode_name] = {
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
            except Exception as e:
                print(f"评估 {mode_name} 时出错: {e}")
                dataset_results[mode_name] = {
                    "correct": 0,
                    "incorrect": 0,
                    "total": 0,
                    "accuracy": 0.0,
                    "description": f"Error: {str(e)}",
                }
        
        all_results[dataset_name] = dataset_results
    
    return all_results


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model_csv_path(model_path):
    """获取模型对应的 CSV 文件路径"""
    model_name = extract_model_name(model_path)
    return os.path.join(OUTPUT_DIR, f"{model_name}.csv")


def save_model_results(model_path: str, all_results: dict):
    """保存模型的所有测试结果到 CSV 文件"""
    csv_path = get_model_csv_path(model_path)
    
    # 准备 CSV 数据
    rows = []
    for dataset_name in DATASETS:
        dataset_name_short = extract_dataset_name(dataset_name)
        if dataset_name_short not in all_results:
            continue
        
        row_data = {'Dataset': dataset_name_short}
        dataset_results = all_results[dataset_name_short]
        
        for metric in METRICS:
            if metric in dataset_results:
                accuracy = dataset_results[metric]['accuracy']
                row_data[metric] = f"{accuracy:.6f}"
            else:
                row_data[metric] = ''
        
        rows.append(row_data)
    
    # 写入 CSV 文件
    headers = ['Dataset'] + METRICS
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        print(f"\n✓ 结果已保存到: {csv_path}")
    except Exception as e:
        print(f"✗ 保存 CSV 文件失败 {csv_path}: {e}")


def main():
    """主函数"""
    print("="*80)
    print("批量测试开始（使用 vLLM serve + 数据并行）")
    print("="*80)
    print(f"模型数量: {len(MODELS)}")
    print(f"数据集数量: {len(DATASETS)}")
    print(f"数据并行大小: {VLLM_DATA_PARALLEL_SIZE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)
    
    # 确保输出目录存在
    ensure_output_dir()
    
    total_models = len(MODELS)
    success_count = 0
    
    start_time = time.time()
    
    for model_idx, model_path in enumerate(MODELS, 1):
        print(f"\n{'#'*80}")
        print(f"处理模型 {model_idx}/{total_models}: {extract_model_name(model_path)}")
        print(f"{'#'*80}")
        
        process = None
        try:
            # 启动服务
            process = start_vllm_service(model_path)
            
            # 等待服务启动
            api_url = f"http://{VLLM_HOST}:{VLLM_PORT}"
            if not wait_for_service(api_url, max_wait=SERVICE_STARTUP_WAIT):
                print(f"✗ 服务启动超时")
                continue
            
            # 测试所有数据集和评估方法
            all_results = test_model_on_all_datasets(model_path, api_url)
            
            # 保存结果
            save_model_results(model_path, all_results)
            success_count += 1
            
        except Exception as e:
            print(f"✗ 处理模型时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 停止服务
            if process is not None:
                stop_vllm_service(process)
                time.sleep(5)  # 等待服务完全关闭
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print("批量测试完成")
    print("="*80)
    print(f"总耗时: {elapsed_time/3600:.2f} 小时 ({elapsed_time/60:.2f} 分钟)")
    print(f"成功: {success_count}/{total_models}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每个模型的结果已保存到独立的 CSV 文件中")
    print("="*80)


if __name__ == "__main__":
    main()

