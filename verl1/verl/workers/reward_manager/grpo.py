# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Tuple, Dict, Any
import logging
import re
import torch
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.utils import compare_latex_expressions
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@register("grpo")
class GRPORewardManager(AbstractRewardManager):
    """GRPO (Group Relative Policy Optimization) Reward Manager.
    
    GRPO computes sequence-level rewards and performs within-group normalization
    to use relative advantages instead of absolute rewards.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        alpha: float = 0.5,
        max_branches: int = 5,
        min_gap: int = 10,
        seg_comp_fn: Optional[Callable[[torch.Tensor, Dict[str, Any]], float]] = None,
        rollout_comp_fn: Optional[Callable[[List[float], Dict[str, Any]], float]] = None,
        **kwargs: Any,
    ):
        """Initialize GRPO Reward Manager.
        
        Args:
            tokenizer: Tokenizer for decoding token IDs
            num_examine: Number of samples to examine (for debugging)
            compute_score: Function to compute reward score (not used in GRPO)
            reward_fn_key: Key to access data source in non-tensor batch
            alpha: Weight parameter (not used in GRPO)
            max_branches: Maximum number of branches for segmentation (not used in GRPO)
            min_gap: Minimum gap for segmentation (not used in GRPO)
            seg_comp_fn: Segment comparison function (not used in GRPO)
            rollout_comp_fn: Rollout comparison function (not used in GRPO)
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.alpha = float(alpha)
        self.max_branches = int(max_branches)
        self.min_gap = int(min_gap)
        self.seg_comp_fn = seg_comp_fn
        self.rollout_comp_fn = rollout_comp_fn

    @staticmethod
    def _default_rollout_comp(ground_truth: str, sequence: str, outLength: int) -> float:
        """Compute rollout-level reward by comparing predicted answer with ground truth.
        
        Args:
            ground_truth: Ground truth answer string
            sequence: Generated sequence string
            outLength: Length of the output sequence
            
        Returns:
            Reward score: 1.0 if correct, -1.0 if incorrect, with length penalty
        """
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, sequence)
        maxLength = 8192
        if len(matches) == 0:
            rw = -1.0
            if outLength > maxLength/2:
                rw -= (outLength)/maxLength
            return rw
        predicted_answer = matches[-1]
        
        if compare_latex_expressions(predicted_answer, ground_truth):
            rw = 1.0
            if outLength > maxLength/2:
                rw -= (outLength)/maxLength
            return rw
        else:
            rw = -1.0
            if outLength > maxLength/2:
                rw -= (outLength)/maxLength
            return rw

    def compute_for_rollout(
        self,
        input_ids: torch.Tensor,
        ground_truth: List[str],
        response_mask: torch.Tensor,
        rollout_outputs: List[torch.Tensor],
        entropys: torch.Tensor,
        repeat_times: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GRPO main entry: compute sequence-level rewards, perform within-group normalization,
        and broadcast to timesteps.
        
        GRPO core idea: normalize sequence rewards within each group (repeat_times samples)
        to use relative advantages instead of absolute rewards.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            ground_truth: List of ground truth answers
            response_mask: Response mask [batch_size, seq_len]
            rollout_outputs: List of token sequences (each is a tensor)
            entropys: Entropy values [batch_size, seq_len]
            repeat_times: Number of samples per group for within-group normalization
            
        Returns:
            Tuple of (rollout_raw_rewards, betaTensor, advTensor):
            - rollout_raw_rewards: Raw sequence rewards broadcast to timesteps [batch_size, seq_len]
            - betaTensor: Normalized within-group relative advantages (sequence-level) [batch_size]
            - advTensor: Per-timestep advantages [batch_size, seq_len]
        """
        per_sample_info: List[Dict[str, Any]] = []
        if rollout_outputs is None:
            return torch.zeros(0), torch.zeros(0), torch.zeros(0)

        # Step 1: Compute sequence-level rewards (GRPO only uses sequence-level rewards)
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
        
        # Initialize all samples' beta to 0 (handle cases where batch size is not divisible by repeat_times)
        for j in range(len(per_sample_info)):
            per_sample_info[j]['beta'] = 0.0
        
        # Normalize within each group (GRPO core: within-group relative advantages)
        # Fix: When rollout count increases, within-group sample count increases,
        # if reward distribution becomes more concentrated, std will decrease.
        # This causes normalized advantage signals to weaken, making training unstable.
        # Solution: Use adaptive minimum std threshold, adjusted based on within-group sample count.
        if repeat_times <= 8:
            min_std = 0.1  # Base minimum std threshold
        elif repeat_times <= 16:
            min_std = 0.15  # Slightly increase for medium rollout count
        else:
            min_std = 0.2  # Further increase for large rollout count
        
        for i in range(len(rollout_raw_reward)//repeat_times):
            rollout_reward = rollout_raw_reward[(i)*repeat_times:(i+1)*repeat_times]
            mean = float(sum(rollout_reward) / len(rollout_reward))
            # Use sample std (unbiased=True) for more accurate variance estimation
            # When within-group sample count increases, using sample std better estimates true variance
            reward_tensor = torch.tensor(rollout_reward, dtype=torch.float32)
            if len(rollout_reward) > 1:
                std = float(torch.std(reward_tensor, unbiased=True))  # Use sample std
            else:
                std = 0.0
            
            # Add adaptive minimum std threshold to prevent advantage signal weakening when rollout count increases
            # When within-group sample count increases, if reward distribution becomes more concentrated,
            # std will decrease. Using adaptive minimum std threshold maintains advantage signal strength.
            std = max(std, min_std)
            
            if std != 0:
                # Compute normalized scores (z-scores)
                tz_scores = (reward_tensor - mean) / std
            else:
                # If std is 0, all samples have same reward, set to 0
                tz_scores = torch.zeros(len(rollout_reward), dtype=torch.float32)
            
            # Assign normalized scores to beta (key fix: use tz_scores instead of raw reward)
            for j in range(i*repeat_times, (i+1)*repeat_times):
                per_sample_info[j]['beta'] = float(tz_scores[j-i*repeat_times].item())
                    
        # Step 2: Build final advantages and broadcast to timesteps
        # GRPO broadcasts normalized within-group relative advantages (beta) to all timesteps
        for idx, info in enumerate(per_sample_info):
            beta = info['beta']
            per_timestep_adv = torch.zeros(len(entropys[idx]), dtype=torch.float32) + beta
            info['per_timestep_adv'] = per_timestep_adv
        print(f"End computing advantages", flush=True)

        # Return normalized beta (within-group relative advantages), not raw reward
        betaTensor = torch.tensor([info['beta'] for info in per_sample_info], dtype=torch.float32)
        advTensor = torch.stack([info['per_timestep_adv'] for info in per_sample_info])
        return rollout_raw_rewards, betaTensor, advTensor

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards using GRPO method.
        
        This method extracts necessary information from DataProto and calls compute_for_rollout.
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return dictionary with extra info
            
        Returns:
            Reward tensor or dictionary with reward tensor and extra info
        """
        # Extract ground truth from non_tensor_batch
        ground_truth = []
        for i in range(len(data)):
            data_item = data[i]
            gt = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            if isinstance(gt, (list, tuple)):
                gt = gt[0] if len(gt) > 0 else ""
            ground_truth.append(str(gt))
        
        # Get rollout outputs (responses)
        rollout_outputs = []
        for i in range(len(data)):
            response_ids = data.batch["responses"][i]
            rollout_outputs.append(response_ids)
        
        # Get entropys if available, otherwise create zeros
        if "entropys" in data.batch:
            entropys = data.batch["entropys"]
        else:
            # Create dummy entropys if not available
            batch_size = len(data)
            seq_len = data.batch["responses"].shape[1]
            entropys = torch.zeros(batch_size, seq_len, dtype=torch.float32)
        
        # Get repeat_times from config or use default
        repeat_times = getattr(self, '_repeat_times', 1)
        
        # Call compute_for_rollout
        rollout_raw_rewards, betaTensor, advTensor = self.compute_for_rollout(
            input_ids=data.batch.get("input_ids", data.batch["prompts"]),
            ground_truth=ground_truth,
            response_mask=data.batch.get("response_mask", torch.ones_like(data.batch["responses"], dtype=torch.bool)),
            rollout_outputs=rollout_outputs,
            entropys=entropys,
            repeat_times=repeat_times
        )
        
        # Use advTensor as reward_tensor for compatibility
        reward_tensor = advTensor
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "beta": betaTensor,
                "advantages": advTensor,
                "rollout_raw_rewards": rollout_raw_rewards,
            }
        else:
            return reward_tensor
