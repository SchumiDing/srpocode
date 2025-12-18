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

    def compute_for_rollout(self, batch: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute GRPO rewards (sequence outcome) without advantages."""
        responses = batch.batch["responses"]
        response_mask = batch.batch.get("response_mask", torch.ones_like(responses, dtype=torch.bool))
        input_ids = batch.batch.get("input_ids", batch.batch["prompts"])

        ground_truth = []
        for i in range(len(batch)):
            data_item = batch[i]
            gt = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            if isinstance(gt, (list, tuple)):
                gt = gt[0] if len(gt) > 0 else ""
            ground_truth.append(str(gt))

        rollout_args = []
        seq_lengths = []
        for i in range(len(responses)):
            sample = responses[i]
            stopidx = sum(response_mask[i])
            outLength = stopidx
            seq_lengths.append(outLength)
            rollout_args.append((ground_truth[i], self.tokenizer.decode(sample[:outLength - 1]), outLength))

        max_workers = min(32, len(rollout_args))
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
                    seqRewards[idx] = -1.0

        token_level_rewards = []
        for idx, (seq_reward, mask_len) in enumerate(zip(seqRewards, seq_lengths)):
            rewards = torch.zeros_like(responses[idx], dtype=torch.float32)
            valid_len = int(max(mask_len - 1, 0))
            if valid_len >= 0:
                rewards[valid_len] = seq_reward
            token_level_rewards.append(rewards)

        reward_tensor = torch.stack(token_level_rewards)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute GRPO rewards; advantages are computed by adv estimator."""
        return self.compute_for_rollout(batch=data, return_dict=return_dict)
