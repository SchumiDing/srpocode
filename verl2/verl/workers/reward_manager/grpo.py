
from typing import Callable, List, Optional, Tuple, Dict, Any
import logging
import re
import torch
from verl.workers.reward_manager import register
endpoint = "http://10.140.37.71:5000/v1/step_rewards"
logger = logging.getLogger(__name__)
proxies = {
    "http": None,
    "https": None
}
from sympy import simplify, parse_expr
from sympy.parsing.latex import parse_latex
from concurrent.futures import ThreadPoolExecutor, as_completed
def compare_latex_expressions(expr1: str, expr2: str) -> bool:
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
@register("grpo")
class GRPORewardManager:
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

    @staticmethod
    def _default_rollout_comp(ground_truth: str, sequence: str, outLength: int) -> float:
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, sequence)
        maxLength = 8192
        if len(matches) == 0:
            # if len(sequence) > 4000:
            #     return -1.0 - len(sequence)/1000
            rw = -1.0
            if outLength > maxLength/2:
                rw -= (outLength)/maxLength
            return rw
        predicted_answer = matches[-1]
        
        if compare_latex_expressions(predicted_answer, ground_truth):
            # print(f"Right answer: \"{predicted_answer}\" == \"{ground_truth}\"", flush=True)
            rw = 1.0
            if outLength > maxLength/2:
                rw -= (outLength)/maxLength
            return rw
        else:
            # print(f"Wrong answer: \"{predicted_answer}\" != \"{ground_truth}\"", flush=True)
            # print(sequence[-100:], flush=True)
            # if len(sequence) > 4000:
            #     return -1.0 - len(sequence)/1000
            rw = -1.0
            if outLength > maxLength/2:
                rw -= (outLength)/maxLength
            return rw

    def compute_for_rollout(self, input_ids,ground_truth, response_mask, rollout_outputs: List[str], entropys: torch.Tensor, repeat_times: int = 1) -> Dict[str, Any]:
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
            rollout_args.append((ground_truth[i], self.tokenizer.decode(sample[:outLength-1]), outLength))
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
        # 修复：当rollout数量增加时，组内样本数量增加，如果奖励分布更集中，标准差会变小
        # 这会导致标准化后的advantage信号变弱，训练不稳定
        # 解决方案：使用自适应最小标准差阈值，根据组内样本数量调整
        # 当rollout数量从8增加到16或32时，组内样本数量增加，奖励分布可能更集中
        # 使用更大的最小标准差阈值可以保持advantage信号的强度
        if repeat_times <= 8:
            min_std = 0.1  # 基础最小标准差阈值
        elif repeat_times <= 16:
            min_std = 0.15  # 中等rollout数量时稍微增加
        else:
            min_std = 0.2  # 大rollout数量时进一步增加
        
        for i in range(len(rollout_raw_reward)//repeat_times):
            # print(f"i: {i}")
            rollout_reward = rollout_raw_reward[(i)*repeat_times:(i+1)*repeat_times]
            mean = float(sum(rollout_reward) / len(rollout_reward))
            # 使用样本标准差（unbiased=True）以获得更准确的方差估计
            # 当组内样本数量增加时，使用样本标准差可以更好地估计真实方差
            reward_tensor = torch.tensor(rollout_reward, dtype=torch.float32)
            if len(rollout_reward) > 1:
                std = float(torch.std(reward_tensor, unbiased=True))  # 使用样本标准差
            else:
                std = 0.0
            
            # 添加自适应最小标准差阈值，防止当rollout数量增加时advantage信号变弱
            # 当组内样本数量增加时，如果奖励分布更集中，标准差会变小
            # 使用自适应最小标准差阈值可以保持advantage信号的强度
            std = max(std, min_std)
            
            if std != 0:
                # 计算标准化分数（z-score）
                tz_scores = (reward_tensor - mean) / std
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
    
    # def __call__(self, data: DataProto, config: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    #     return self.compute_for_rollout(data.batch)