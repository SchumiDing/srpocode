from typing import Callable, List, Optional, Tuple, Dict, Any
import logging
import re
import torch
import os
import tempfile
import subprocess
import json
from verl.workers.reward_manager import register
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def extract_code_from_tags(sequence: str) -> Optional[str]:
    """提取 <code> </code> 标签包裹的代码"""
    pattern = r'<code>(.*?)</code>'
    matches = re.findall(pattern, sequence, re.DOTALL)
    if matches:
        return matches[-1]  # 返回最后一个匹配的代码块
    return None


def extract_largest_json(text: str) -> Optional[Dict]:
    """提取文本中最大的 JSON 对象（{} 包裹的部分）"""
    # 找到所有可能的 JSON 对象
    json_candidates = []
    depth = 0
    start = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                json_str = text[start:i+1]
                try:
                    json_obj = json.loads(json_str)
                    json_candidates.append((len(json_str), json_obj))
                except json.JSONDecodeError:
                    pass
                start = -1
    
    if json_candidates:
        # 返回最大的 JSON 对象
        largest = max(json_candidates, key=lambda x: x[0])
        return largest[1]
    return None


def compare_answers(predicted: Any, ground_truth: str) -> bool:
    """比较答案，允许千分之一数量级的误差"""
    try:
        # 尝试将 ground_truth 转换为数值
        gt_value = float(ground_truth)
        
        # 尝试将 predicted 转换为数值
        if isinstance(predicted, str):
            pred_value = float(predicted)
        elif isinstance(predicted, (int, float)):
            pred_value = float(predicted)
        else:
            return False
        
        # 计算相对误差（允许千分之一数量级的误差）
        if abs(gt_value) < 1e-10:  # 处理接近0的情况
            return abs(pred_value) < 1e-10
        
        relative_error = abs(pred_value - gt_value) / abs(gt_value)
        return relative_error <= 0.001  # 千分之一
        
    except (ValueError, TypeError):
        # 如果无法转换为数值，进行字符串比较
        return str(predicted).strip() == str(ground_truth).strip()


@register("orlm")
class ORLMRewardManager:
    def __init__(self,
                 tokenizer: Any,
                 num_examine: int,
                 compute_score,
                 reward_fn_key: str,
                 alpha: float = 0.5,
                 max_branches: int = 5,
                 min_gap: int = 10,
                 seg_comp_fn: Optional[Callable[[torch.Tensor, Dict[str, Any]], float]] = None,
                 rollout_comp_fn: Optional[Callable[[List[float], Dict[str, Any]], float]] = None):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.alpha = float(alpha)
        self.max_branches = int(max_branches)
        self.min_gap = int(min_gap)
        self.seg_comp_fn = seg_comp_fn
        self.rollout_comp_fn = rollout_comp_fn
        self.tokenizer = tokenizer
        # 创建临时目录用于存储代码文件
        self.temp_dir = tempfile.mkdtemp(prefix="orlm_code_")
        logger.info(f"Created temporary directory for code execution: {self.temp_dir}")

    @staticmethod
    def _default_rollout_comp(ground_truth: str, sequence: str, outLength: int, temp_dir: str) -> float:
        """
        计算序列级别奖励：
        1. 提取 <code> </code> 包裹的代码
        2. 保存到临时文件并运行
        3. 如果运行报错，返回 -2
        4. 提取输出中最大的 JSON
        5. 检查 JSON 中的 "answer" key
        6. 比较答案：正确返回 1，错误或缺失返回 -1
        """
        # 提取代码
        code = extract_code_from_tags(sequence)
        if code is None:
            return -1.0
        
        # 创建临时文件保存代码
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=temp_dir, delete=False) as code_file:
            code_file.write(code)
            code_path = code_file.name
        
        # 创建输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.out', dir=temp_dir, delete=False) as out_file:
            out_path = out_file.name
        
        try:
            # 运行代码，重定向输出到文件
            result = subprocess.run(
                ['python', code_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,  # 30秒超时
                text=True
            )
            
            # 如果运行报错，返回 -2
            if result.returncode != 0:
                # 清理临时文件
                try:
                    os.unlink(code_path)
                    os.unlink(out_path)
                except:
                    pass
                return -2.0
            
            # 获取输出
            output = result.stdout
            
            # 提取最大的 JSON
            json_obj = extract_largest_json(output)
            if json_obj is None:
                # 清理临时文件
                try:
                    os.unlink(code_path)
                    os.unlink(out_path)
                except:
                    pass
                return -1.0
            
            # 检查是否有 "answer" key
            if "answer" not in json_obj:
                # 清理临时文件
                try:
                    os.unlink(code_path)
                    os.unlink(out_path)
                except:
                    pass
                return -1.0
            
            # 比较答案
            predicted_answer = json_obj["answer"]
            if compare_answers(predicted_answer, ground_truth):
                # 清理临时文件
                try:
                    os.unlink(code_path)
                    os.unlink(out_path)
                except:
                    pass
                return 1.0
            else:
                # 清理临时文件
                try:
                    os.unlink(code_path)
                    os.unlink(out_path)
                except:
                    pass
                return -1.0
                
        except subprocess.TimeoutExpired:
            # 超时，返回 -2
            try:
                os.unlink(code_path)
                os.unlink(out_path)
            except:
                pass
            return -2.0
        except Exception as e:
            # 其他异常，返回 -2
            logger.warning(f"Error executing code: {e}")
            try:
                os.unlink(code_path)
                os.unlink(out_path)
            except:
                pass
            return -2.0

    def compute_for_rollout(self, input_ids, ground_truth, response_mask, rollout_outputs: List[str], entropys: torch.Tensor, repeat_times: int = 1) -> Dict[str, Any]:
        """
        GRPO主入口：计算序列级别奖励，进行组内标准化（组内相对优势），并广播到时间步。
        GRPO的核心思想：对同一组（repeat_times个样本）内的序列奖励进行标准化，使用相对优势而非绝对奖励。
        Args:
            rollout_outputs: list of token sequences
            repeat_times: 每组内的样本数量，用于组内标准化
        Returns:
            rollout_raw_rewards: 原始序列奖励（广播到时间步）
            betaTensor: 标准化后的组内相对优势（序列级别）
            advTensor: 每个时间步的advantage（等于beta，因为GRPO只使用序列级别奖励）
        """
        per_sample_info: List[Dict[str, Any]] = []
        if rollout_outputs is None:
            return {'per_sample': per_sample_info}

        # 第一步：计算序列级别奖励（GRPO只使用序列级别奖励）
        print(f"Start computing rewards", flush=True)
        rollout_args = []
        for i in range(len(rollout_outputs)):
            sample = rollout_outputs[i]
            stopidx = sum(response_mask[i])
            outLength = stopidx
            rollout_args.append((ground_truth[i], self.tokenizer.decode(sample[:outLength-1]), outLength, self.temp_dir))
            per_sample_info.append({
                'T': len(entropys[i])
            })
        print(f"Start computing rollout rewards", flush=True)
        # Use ThreadPoolExecutor instead of multiprocessing.Pool to avoid deadlock in Ray environment
        # Ray manages its own processes, so creating subprocess pools can cause conflicts
        max_workers = min(32, len(rollout_args))  # Limit to 32 workers to avoid resource exhaustion
        seqRewards = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self._default_rollout_comp, *args): i for i, args in enumerate(rollout_args)}
            seqRewards = [None] * len(rollout_args)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    seqRewards[idx] = future.result()
                except Exception as e:
                    print(f"[WARNING] Error computing reward for sample {idx}: {e}", flush=True)
                    seqRewards[idx] = -1.0  # Default to negative reward on error
        print(f"End computing rollout rewards", flush=True)
        for i, seqReward in enumerate(seqRewards):
            per_sample_info[i]['seq_reward'] = seqReward

        rollout_raw_rewards = [info['seq_reward']+torch.zeros(len(entropys[idx])) for idx, info in enumerate(per_sample_info)]
        rollout_raw_rewards = torch.stack(rollout_raw_rewards)
        rollout_raw_reward = [info['seq_reward'] for info in per_sample_info]
        print(f"Start computing advantages", flush=True)
        
        # 初始化所有样本的beta为0（处理不能被repeat_times整除的情况）
        for j in range(len(per_sample_info)):
            per_sample_info[j]['beta'] = 0.0
        
        # 对每个组进行标准化（GRPO的核心：组内相对优势）
        for i in range(len(rollout_raw_reward)//repeat_times):
            rollout_reward = rollout_raw_reward[(i)*repeat_times:(i+1)*repeat_times]
            mean = float(sum(rollout_reward) / len(rollout_reward))
            std = float(torch.std(torch.tensor(rollout_reward, dtype=torch.float32)))
            
            if std != 0:
                # 计算标准化分数（z-score）
                tz_scores = (torch.tensor(rollout_reward, dtype=torch.float32) - mean) / std
            else:
                # 如果std为0，所有样本reward相同，设为0
                tz_scores = torch.zeros(len(rollout_reward), dtype=torch.float32)
            
            # 将标准化后的分数赋值给beta（关键修复：使用tz_scores而不是原始reward）
            for j in range(i*repeat_times, (i+1)*repeat_times):
                per_sample_info[j]['beta'] = float(tz_scores[j-i*repeat_times].item())
                    
        # 第二步：构建 final advantages 并广播到时间步
        # GRPO将标准化后的组内相对优势（beta）广播到所有时间步
        for idx, info in enumerate(per_sample_info):
            beta = info['beta']
            per_timestep_adv = torch.zeros(len(entropys[idx]), dtype=torch.float32) + beta
            info['per_timestep_adv'] = per_timestep_adv
        print(f"End computing advantages", flush=True)

        # 返回标准化后的beta（组内相对优势），而不是原始reward
        betaTensor = torch.tensor([info['beta'] for info in per_sample_info], dtype=torch.float32)
        advTensor = torch.stack([info['per_timestep_adv'] for info in per_sample_info])
        return rollout_raw_rewards, betaTensor, advTensor

