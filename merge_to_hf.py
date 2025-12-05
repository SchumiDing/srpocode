#!/usr/bin/env python3
"""
合并FSDP checkpoint并生成HuggingFace格式的模型（safetensors格式）
包含tokenizer和chat template等信息
"""

import argparse
import os
import shutil
from pathlib import Path

from model_merger import FSDPModelMerger, ModelMergerConfig


def merge_checkpoint_to_huggingface(
    checkpoint_dir: str,
    output_dir: str,
    use_safetensors: bool = True,
    skip_test: bool = False,
    test_device: str = "cpu",
):
    """
    合并FSDP checkpoint并保存为HuggingFace格式
    
    Args:
        checkpoint_dir: checkpoint目录路径（包含model_world_size_*_rank_*.pt文件）
        output_dir: 输出目录路径
        use_safetensors: 是否使用safetensors格式（默认True）
        skip_test: 是否跳过模型测试（默认False）
        test_device: 测试时使用的设备（默认"cpu"）
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    
    # 检查checkpoint目录
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 检查huggingface子目录
    hf_config_dir = checkpoint_dir / "huggingface"
    if not hf_config_dir.exists():
        print(f"Warning: huggingface subdirectory not found in {checkpoint_dir}")
        print("Will use checkpoint_dir as config path")
        hf_config_dir = checkpoint_dir
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"HuggingFace config directory: {hf_config_dir}")
    print(f"Output directory: {output_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置ModelMerger
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(checkpoint_dir),
        hf_model_config_path=str(hf_config_dir),
        target_dir=str(output_dir),
        hf_upload_path=None,
        private=False,
        tie_word_embedding=False,
        is_value_model=False,
        hf_model_path=None,
    )
    
    # 创建merger并执行合并
    print("\n" + "="*80)
    print("Starting model merge...")
    print("="*80)
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    
    # 确保使用safetensors格式
    if use_safetensors:
        print("\n" + "="*80)
        print("Converting to safetensors format...")
        print("="*80)
        _ensure_safetensors(output_dir)
    
    # 复制huggingface目录中的额外文件（如chat_template.jinja等）
    print("\n" + "="*80)
    print("Copying additional files from huggingface directory...")
    print("="*80)
    _copy_additional_files(hf_config_dir, output_dir)
    
    print("\n" + "="*80)
    print(f"✅ Successfully merged checkpoint to HuggingFace format!")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # # 运行测试
    # if not skip_test:
    #     print("\n" + "="*80)
    #     print("Running test questions...")
    #     print("="*80)
    #     test_merged_model(str(output_dir), device=test_device)
    # else:
    #     print("\nSkipping model test (--skip-test specified)")


def _ensure_safetensors(output_dir: Path):
    """确保模型使用safetensors格式，如果存在pytorch_model.bin则转换"""
    import torch
    from safetensors.torch import save_file
    
    pytorch_model_bin = output_dir / "pytorch_model.bin"
    safetensors_files = list(output_dir.glob("model*.safetensors"))
    
    # 检查是否已经是safetensors格式
    if safetensors_files and not pytorch_model_bin.exists():
        print(f"✅ Model already in safetensors format: {len(safetensors_files)} files found")
        return
    
    # 如果存在pytorch_model.bin，转换为safetensors
    if pytorch_model_bin.exists():
        print(f"Loading {pytorch_model_bin}...")
        state_dict = torch.load(pytorch_model_bin, map_location="cpu", weights_only=False)
        
        # 计算总大小
        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        print(f"Model size: {total_size / 1024**3:.2f} GB")
        
        # 使用transformers的默认分片策略（5GB per shard）
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
        
        if total_size > max_shard_size:
            # 需要分片
            print(f"Model size exceeds {max_shard_size / 1024**3:.2f} GB, splitting into multiple files...")
            _convert_to_sharded_safetensors(state_dict, output_dir, max_shard_size)
        else:
            # 单个文件
            safetensors_file = output_dir / "model.safetensors"
            print(f"Saving to {safetensors_file}...")
            save_file(state_dict, safetensors_file)
            print(f"✅ Saved {safetensors_file} ({total_size / 1024**3:.2f} GB)")
        
        # 删除pytorch_model.bin
        pytorch_model_bin.unlink()
        print(f"Removed {pytorch_model_bin}")
    else:
        print("⚠️  Warning: No pytorch_model.bin or safetensors files found")


def _convert_to_sharded_safetensors(state_dict: dict, output_dir: Path, max_shard_size: int):
    """将state_dict转换为分片的safetensors格式"""
    import json
    from safetensors.torch import save_file
    
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    shard_size = 0
    shard_index = 0
    current_shard = {}
    weight_map = {}
    
    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        
        # 如果当前shard加上这个tensor会超过限制，且当前shard不为空，则保存当前shard
        if shard_size + tensor_size > max_shard_size and current_shard:
            shard_filename = f"model-{shard_index:05d}-of-00001.safetensors"  # 先占位，后面会更新
            shard_path = output_dir / shard_filename
            save_file(current_shard, shard_path)
            print(f"Saved shard {shard_index}: {shard_filename} ({shard_size / 1024**3:.2f} GB)")
            
            for k in current_shard.keys():
                weight_map[k] = shard_filename
            
            current_shard = {}
            shard_size = 0
            shard_index += 1
        
        current_shard[key] = tensor
        shard_size += tensor_size
    
    # 保存最后一个shard
    if current_shard:
        total_shards = shard_index + 1
        shard_filename = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        shard_path = output_dir / shard_filename
        save_file(current_shard, shard_path)
        print(f"Saved shard {shard_index}: {shard_filename} ({shard_size / 1024**3:.2f} GB)")
        
        for k in current_shard.keys():
            weight_map[k] = shard_filename
        
        # 更新之前所有shard的文件名
        for i in range(shard_index):
            old_filename = f"model-{i:05d}-of-00001.safetensors"
            new_filename = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old_path = output_dir / old_filename
            new_path = output_dir / new_filename
            if old_path.exists():
                old_path.rename(new_path)
                # 更新weight_map中对应的文件名
                for k, v in weight_map.items():
                    if v == old_filename:
                        weight_map[k] = new_filename
        
        # 创建index文件
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        index_file = output_dir / "model.safetensors.index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"✅ Created index file: {index_file}")
        print(f"Total shards: {total_shards}")


def test_merged_model(model_path: str, device: str = "cpu"):
    """
    测试合并后的模型，运行两个测试问题
    
    Args:
        model_path: 合并后的模型路径
        device: 运行设备（默认"cpu"）
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        ).eval()
        
        print("✅ Model loaded successfully!")
        
        # 测试问题1
        print("\n" + "-"*80)
        print("Test Question 1: Triangle Problem")
        print("-"*80)
        prompt1 = """\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\n\nLet's think step by step and put the final answer in the \\boxed{} tag. Do not repeat any sentences in the answer, and keep only one \\boxed{} tag which contains the final answer."""
        ans1 = "34"
        
        messages1 = [{"role": "user", "content": prompt1}]
        text1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        
        print("\nInput prompt:")
        print(text1[:200] + "..." if len(text1) > 200 else text1)
        
        inputs1 = tokenizer(text1, return_tensors="pt").to(model.device)
        outputs1 = model.generate(**inputs1, max_new_tokens=2048)
        result1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
        
        print("\nGenerated output:")
        print(result1)
        print(f"\nExpected answer: {ans1}")
        
        # 测试问题2
        print("\n" + "-"*80)
        print("Test Question 2: Right Triangle Problem")
        print("-"*80)
        prompt2 = """\n Given a right triangle with two legs of length 3 and 4, find the length of the hypotenuse.

Let's think step by step and put the final answer in the \\boxed{} tag. Do not repeat any sentences in the answer, and keep only one \\boxed{} tag which contains the final answer."""
        ans2 = "5"
        
        messages2 = [{"role": "user", "content": prompt2}]
        text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        
        print("\nInput prompt:")
        print(text2[:200] + "..." if len(text2) > 200 else text2)
        
        inputs2 = tokenizer(text2, return_tensors="pt").to(model.device)
        outputs2 = model.generate(**inputs2, max_new_tokens=2048)
        result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
        
        print("\nGenerated output:")
        print(result2)
        print(f"\nExpected answer: {ans2}")
        
        print("\n" + "="*80)
        print("✅ Test completed!")
        print("="*80)
        
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to test model: {e}")
        print("You can test the model manually using testmodel.py")
        import traceback
        traceback.print_exc()


def _copy_additional_files(hf_config_dir: Path, output_dir: Path):
    """复制huggingface目录中的额外文件"""
    files_to_copy = [
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]
    
    copied_count = 0
    for filename in files_to_copy:
        src_file = hf_config_dir / filename
        if src_file.exists():
            dst_file = output_dir / filename
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {filename}")
            copied_count += 1
        else:
            print(f"Skipped (not found): {filename}")
    
    print(f"\nCopied {copied_count} additional files")


def main():
    parser = argparse.ArgumentParser(
        description="合并FSDP checkpoint并生成HuggingFace格式的模型（safetensors格式）"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint目录路径（包含model_world_size_*_rank_*.pt文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径（HuggingFace模型将保存到这里）"
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_true",
        help="不使用safetensors格式（保留pytorch_model.bin）"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="跳过模型测试（默认会在合并后运行测试）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="测试时使用的设备（默认cpu，可以是cuda:0等）"
    )
    
    args = parser.parse_args()
    
    merge_checkpoint_to_huggingface(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        use_safetensors=not args.no_safetensors,
        skip_test=args.skip_test,
        test_device=args.device,
    )


if __name__ == "__main__":
    main()

