#!/usr/bin/env python3
"""
批量评估checkpoint目录下的所有checkpoint
- 合并FSDP checkpoint到HuggingFace格式
- 对每个checkpoint进行测试
- 保存准确率结果到CSV
- 清理临时文件
"""

import argparse
import os
import shutil
import re
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sympy import simplify
from sympy.parsing.latex import parse_latex
from merge_to_hf import merge_checkpoint_to_huggingface

torch.manual_seed(42)
# 数据集路径映射
DATASET_PATHS = {
    "math500": "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/math500split_val.parquet",
    "gsm8k": "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/gsm8k/gsm8ktest.parquet",
    "aime2024": "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2024.parquet",
    "aime2025": "/mnt/shared-storage-user/mineru4s/dingruiyi/srpo/data/aime2025.parquet",
}

# Greedy解码的数据集
GREEDY_DATASETS = {"math500", "gsm8k"}


def compare_latex_expressions(expr1: str, expr2: str) -> bool:
    """比较两个LaTeX表达式是否相等"""
    if expr1.strip() == expr2.strip():
        return True
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


def evaluate_model(model_path: str, val_data_path: str, use_greedy: bool = True, 
                   batch_size: int = 4, num_seqs: int = 32, temperature: float = 0.7, 
                   top_p: float = 0.5) -> dict:
    """
    评估模型在给定数据集上的准确率
    
    Args:
        model_path: 模型路径
        val_data_path: 验证数据路径
        use_greedy: 是否使用greedy解码
        batch_size: batch大小
        num_seqs: 采样时的序列数量（仅当use_greedy=False时使用）
        temperature: 采样温度（仅当use_greedy=False时使用）
        top_p: top_p采样参数（仅当use_greedy=False时使用）
    
    Returns:
        包含准确率信息的字典
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    # 加载数据
    df = pd.read_parquet(val_data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    dataset = MathDataset(df, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
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
            
            if use_greedy:
                # Greedy解码
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                num_seqs_actual = 1
            else:
                # 采样解码
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=num_seqs
                )
                num_seqs_actual = num_seqs
            
            # 逐条解码
            batch_size_actual = len(gts)
            
            for b, gt in enumerate(gts):
                # 获取当前问题的所有回答序列
                start_idx = b * num_seqs_actual
                end_idx = (b + 1) * num_seqs_actual
                question_outputs = outputs[start_idx:end_idx]
                
                # 检查所有回答中是否有任何一个正确
                found_correct = False
                for seq_idx in range(num_seqs_actual):
                    resp = tokenizer.decode(question_outputs[seq_idx], skip_special_tokens=True)
                    matches = re.findall(pattern, resp)
                    if not matches:
                        continue
                    ans = matches[-1]
                    if compare_latex_expressions(ans, gt):
                        found_correct = True
                        break
                
                # 对于每个问题，只要有一个回答正确就算正确
                if found_correct:
                    correct += 1
                else:
                    incorrect += 1
            
            # 更新进度条
            total = correct + incorrect
            pbar.set_postfix({
                'correct': correct,
                'incorrect': incorrect,
                'accuracy': f"{correct/total*100:.2f}%" if total else "0.00%"
            })
    
    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0.0
    
    # 清理GPU内存
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return {
        'correct': correct,
        'incorrect': incorrect,
        'total': total,
        'accuracy': accuracy
    }


def find_all_checkpoints(checkpoint_dir: Path) -> list:
    """找到所有checkpoint目录"""
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith('global_step_'):
            checkpoints.append(item)
    # 按step排序
    checkpoints.sort(key=lambda x: int(x.name.split('_')[-1]))
    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="批量评估checkpoint目录下的所有checkpoint"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="包含多个checkpoint的目录路径（如 math500-qwen2.5-math-1.5b-rflux-r8）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["math500", "gsm8k", "aime2024", "aime2025"],
        help="测试数据集名称"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录（默认在checkpoint_dir同级目录）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch大小（默认4）"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="临时文件目录（默认在checkpoint_dir同级目录的temp_eval目录）"
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_eval_results"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定临时目录
    if args.temp_dir is None:
        temp_dir = checkpoint_dir.parent / f"temp_eval_{checkpoint_dir.name}"
    else:
        temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据集路径
    val_data_path = DATASET_PATHS[args.dataset]
    if not Path(val_data_path).exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")
    
    # 确定是否使用greedy解码
    use_greedy = True
    
    # 找到所有checkpoint
    checkpoints = find_all_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    print(f"Dataset: {args.dataset} (use_greedy={use_greedy})")
    print(f"Output directory: {output_dir}")
    print(f"Temp directory: {temp_dir}")
    print("="*80)
    
    # 存储结果
    results = []
    
    # 遍历每个checkpoint
    for ckpt_path in tqdm(checkpoints, desc="Processing checkpoints"):
        step = ckpt_path.name.split('_')[-1]
        actor_dir = ckpt_path / "actor"
        
        if not actor_dir.exists():
            print(f"Warning: actor directory not found in {ckpt_path}, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing checkpoint: {ckpt_path.name}")
        print(f"{'='*80}")
        
        # 合并checkpoint到临时目录
        merged_model_dir = temp_dir / f"merged_{step}"
        try:
            print(f"Merging checkpoint to {merged_model_dir}...")
            merge_checkpoint_to_huggingface(
                checkpoint_dir=str(actor_dir),
                output_dir=str(merged_model_dir),
                use_safetensors=True,
                skip_test=True
            )
            print(f"✅ Checkpoint merged successfully")
        except Exception as e:
            print(f"❌ Failed to merge checkpoint: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'checkpoint': ckpt_path.name,
                'step': step,
                'accuracy': None,
                'correct': None,
                'incorrect': None,
                'total': None,
                'error': str(e)
            })
            continue
        
        # 评估模型
        try:
            print(f"Evaluating model...")
            if use_greedy:
                eval_result = evaluate_model(
                    model_path=str(merged_model_dir),
                    val_data_path=val_data_path,
                    use_greedy=True,
                    batch_size=args.batch_size,
                    num_seqs=1,
                    temperature=None,
                    top_p=None
                )
            else:
                eval_result = evaluate_model(
                    model_path=str(merged_model_dir),
                    val_data_path=val_data_path,
                    use_greedy=False,
                    batch_size=args.batch_size,
                    num_seqs=32,
                    temperature=0.7,
                    top_p=0.5
                )
            
            print(f"✅ Evaluation completed: Accuracy = {eval_result['accuracy']:.4f} ({eval_result['accuracy']*100:.2f}%)")
            
            results.append({
                'checkpoint': ckpt_path.name,
                'step': step,
                'accuracy': eval_result['accuracy'],
                'correct': eval_result['correct'],
                'incorrect': eval_result['incorrect'],
                'total': eval_result['total'],
                'error': None
            })
        except Exception as e:
            print(f"❌ Failed to evaluate model: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'checkpoint': ckpt_path.name,
                'step': step,
                'accuracy': None,
                'correct': None,
                'incorrect': None,
                'total': None,
                'error': str(e)
            })
        
        # 删除临时合并的模型
        try:
            print(f"Cleaning up temporary model directory...")
            if merged_model_dir.exists():
                shutil.rmtree(merged_model_dir)
                print(f"✅ Temporary directory removed")
        except Exception as e:
            print(f"⚠️  Warning: Failed to remove temporary directory: {e}")
    
    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    output_csv = output_dir / f"eval_results_{args.dataset}.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"✅ All evaluations completed!")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*80}")
    
    # 打印摘要
    print("\nSummary:")
    print(results_df.to_string(index=False))
    
    # 清理临时目录
    try:
        if temp_dir.exists() and any(temp_dir.iterdir()):
            print(f"\nCleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)
            print(f"✅ Temp directory removed")
    except Exception as e:
        print(f"⚠️  Warning: Failed to remove temp directory: {e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: No GPU found. Evaluation may be slow.")
    main()

