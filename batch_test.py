#!/usr/bin/env python3
"""
批量测试脚本：对所有模型和数据集组合进行测试，每个模型的结果保存到独立的 CSV 文件
"""
import os
import json
import subprocess
import tempfile
from pathlib import Path
import csv
import time

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
OUTPUT_DIR = "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/batch_evaluation_results"

# 测试指标列表
METRICS = ["pass@1", "pass@8", "pass@16", "pass@32", "pass@64", "pass@128", "pass@256", "mean32"]

# vLLM 配置
VLLM_TP_SIZE = 4       # 张量并行卡数
VLLM_BATCH_SIZE = 128   # vLLM 侧批量大小，可按显存调大
VLLM_MAX_NEW_TOKENS = 2048  # 控制生成长度，减少显存与耗时


def extract_model_name(model_path):
    """从模型路径中提取模型名称"""
    return os.path.basename(model_path.rstrip('/'))


def extract_dataset_name(dataset_path):
    """从数据集路径中提取数据集名称"""
    filename = os.path.basename(dataset_path)
    return os.path.splitext(filename)[0]


def run_test(model_path, dataset_path):
    """运行单个测试并返回结果"""
    print(f"\n{'='*80}")
    print(f"测试: {extract_model_name(model_path)} - {extract_dataset_name(dataset_path)}")
    print(f"{'='*80}")
    
    # 创建临时 JSON 文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        # 运行测试
        cmd = [
            "python", "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/naivetest_vllm.py",
            "--model_path", model_path,
            "--dataset_path", dataset_path,
            "--test_mode", "all",
            "--output_json", json_path,
            "--tp_size", str(VLLM_TP_SIZE),
            "--batch_size", str(VLLM_BATCH_SIZE),
            "--max_new_tokens", str(VLLM_MAX_NEW_TOKENS),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"错误: 测试失败")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # 读取结果
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get('results', {})
        else:
            print(f"警告: JSON 文件不存在: {json_path}")
            return None
            
    except Exception as e:
        print(f"异常: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(json_path):
            os.remove(json_path)


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model_csv_path(model_path):
    """获取模型对应的 CSV 文件路径"""
    model_name = extract_model_name(model_path)
    return os.path.join(OUTPUT_DIR, f"{model_name}.csv")


def load_model_csv(model_path):
    """加载模型 CSV 文件，返回数据集名称到数据行的映射"""
    csv_path = get_model_csv_path(model_path)
    if not os.path.exists(csv_path):
        return {}
    
    dataset_data = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_name = row['Dataset']
                dataset_data[dataset_name] = row
    except Exception as e:
        print(f"警告: 读取 CSV 文件失败 {csv_path}: {e}")
    
    return dataset_data


def save_model_csv(model_path, dataset_path, results):
    """保存单个模型-数据集组合的测试结果到 CSV 文件"""
    csv_path = get_model_csv_path(model_path)
    dataset_name = extract_dataset_name(dataset_path)
    
    # 加载现有数据
    dataset_data = load_model_csv(model_path)
    
    # 更新当前数据集的结果
    row_data = {'Dataset': dataset_name}
    for metric in METRICS:
        if metric in results:
            accuracy = results[metric]['accuracy']
            row_data[metric] = f"{accuracy:.6f}"  # 保存为小数格式
        else:
            row_data[metric] = ''
    
    dataset_data[dataset_name] = row_data
    
    # 写入 CSV 文件
    headers = ['Dataset'] + METRICS
    
    # 确保数据集按原始顺序写入
    all_datasets = [extract_dataset_name(ds) for ds in DATASETS]
    
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            # 先写入已测试的数据集（按原始顺序）
            for ds_name in all_datasets:
                if ds_name in dataset_data:
                    writer.writerow(dataset_data[ds_name])
            
            # 再写入其他数据集（如果有）
            for ds_name, row in dataset_data.items():
                if ds_name not in all_datasets:
                    writer.writerow(row)
        
        print(f"✓ 已保存到: {csv_path}")
    except Exception as e:
        print(f"✗ 保存 CSV 文件失败 {csv_path}: {e}")


def main():
    """主函数"""
    print("="*80)
    print("批量测试开始")
    print("="*80)
    print(f"模型数量: {len(MODELS)}")
    print(f"数据集数量: {len(DATASETS)}")
    print(f"总测试数: {len(MODELS) * len(DATASETS)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)
    
    # 确保输出目录存在
    ensure_output_dir()
    
    total_tests = len(MODELS) * len(DATASETS)
    current_test = 0
    success_count = 0
    
    start_time = time.time()
    
    for model_path in MODELS:
        for dataset_path in DATASETS:
            current_test += 1
            print(f"\n进度: {current_test}/{total_tests}")
            
            results = run_test(model_path, dataset_path)
            
            if results:
                # 立即保存到对应模型的 CSV 文件
                save_model_csv(model_path, dataset_path, results)
                success_count += 1
                print(f"✓ 测试完成")
            else:
                print(f"✗ 测试失败")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print("批量测试完成")
    print("="*80)
    print(f"总耗时: {elapsed_time/3600:.2f} 小时 ({elapsed_time/60:.2f} 分钟)")
    print(f"成功: {success_count}/{total_tests}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每个模型的结果已保存到独立的 CSV 文件中")
    print("="*80)


if __name__ == "__main__":
    main()

