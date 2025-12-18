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
import torch.nn.functional as F
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.utils import (
    compare_latex_expressions,
    parse_llama3_conversation,
    parse_qwen_conversation,
    parse_deepseek_conversation,
)
import requests
import time
import multiprocessing as mp

logger = logging.getLogger(__name__)


def entropy_segmentation(entropies: torch.Tensor, start_idx: int, outLength: int, max_branches: int = 5, min_gap: int = 10) -> List[Tuple[int, int]]:
    """Segment sequence based on entropies, return list of (start, end) intervals.
    
    Args:
        entropies: 1D tensor, length T
        start_idx: Start index for segmentation
        outLength: Output length
        max_branches: Maximum number of candidate anchors (select top-k high entropy points as anchors)
        min_gap: Minimum distance between two anchors, also requires anchor to be at least min_gap from sequence edges
        
    Returns:
        List[(start, end)] covering [0, T)
    """
    if outLength-start_idx < max_branches+1:
        return [(start_idx, outLength)]
    # Find max_branches anchors starting from start_idx, remove anchors with spacing less than min_gap
    topk_idx = torch.topk(entropies[start_idx:outLength], k=max_branches).indices
    topk_id = topk_idx + start_idx
    topk_id = sorted(topk_id)
    topk_idx = []
    n = outLength
    for i in range(len(topk_id)-1):
        j = i+1
        if topk_id[j] - topk_id[i] > min_gap:
            topk_idx.append(topk_id[i])
        else:
            while j<len(topk_id) and topk_id[j] - topk_id[i] <= min_gap:
                j += 1
            i = j
            if i < len(topk_id):
                topk_idx.append(topk_id[i])
    cuts = []
    last_cut = start_idx
    # Treat each anchor as a cut point (segments form between points)
    for a in topk_idx:
        if a - last_cut >= min_gap:
            cuts.append(a)
            last_cut = a
    # Build segments
    segments: List[Tuple[int, int]] = []
    if len(cuts) == 0:
        segments = [(0, n)]
    else:
        prev = 0
        for c in cuts:
            segments.append((prev, c))
            prev = c
        if prev < n:
            segments.append((prev, n))

    sanitized: List[Tuple[int, int]] = []
    cur = 0
    for (s, e) in segments:
        s = max(0, min(n, int(s)))
        e = max(0, min(n, int(e)))
        if e <= s:
            continue
        if s > cur:
            sanitized.append((cur, s))
        sanitized.append((s, e))
        cur = e
    if cur < n:
        sanitized.append((cur, n))

    if len(sanitized) == 0:
        return [(0, n)]
    return sanitized


@register("rflux")
class RfluxRewardManager(AbstractRewardManager):
    """RFLUX Reward Manager.
    
    RFLUX combines sequence reward with mean segment rewards, uses only beta as advantage.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        alpha: float = 1.0,
        max_branches: int = 5,
        min_gap: int = 10,
        seg_comp_fn: Optional[Callable[[torch.Tensor, Dict[str, Any]], float]] = None,
        rollout_comp_fn: Optional[Callable[[List[float], Dict[str, Any]], float]] = None,
        endpoint: Optional[str] = None,
        proxies: Optional[Dict[str, Optional[str]]] = None,
        conversation_parser: str = "qwen",  # "qwen", "llama3", "deepseek", or "auto"
        **kwargs: Any,
    ):
        """Initialize SRPO Reward Manager.
        
        Args:
            tokenizer: Tokenizer for decoding token IDs
            num_examine: Number of samples to examine (for debugging)
            compute_score: Function to compute reward score
            reward_fn_key: Key to access data source in non-tensor batch
            alpha: Weight parameter (default 1.0)
            max_branches: Maximum number of branches for segmentation
            min_gap: Minimum gap for segmentation
            seg_comp_fn: Segment comparison function
            rollout_comp_fn: Rollout comparison function
            endpoint: Optional endpoint URL for segment reward computation
            proxies: Optional proxy configuration for HTTP requests
            conversation_parser: Which conversation parser to use ("qwen", "llama3", "deepseek", or "auto")
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.alpha = 1.0
        self.max_branches = int(max_branches)
        self.min_gap = int(min_gap)
        self.seg_comp_fn = seg_comp_fn
        self.rollout_comp_fn = rollout_comp_fn
        self.endpoint = endpoint
        self.proxies = proxies or {"http": None, "https": None}
        self.conversation_parser = conversation_parser
        self.scv = torch.sqrt(torch.tensor(1/24))
        
        # Select conversation parser function
        if conversation_parser == "qwen":
            self.parse_conversation = parse_qwen_conversation
        elif conversation_parser == "llama3":
            self.parse_conversation = parse_llama3_conversation
        elif conversation_parser == "deepseek":
            self.parse_conversation = parse_deepseek_conversation
        elif conversation_parser == "auto":
            # Auto-detect based on tokenizer or try multiple parsers
            self.parse_conversation = self._auto_parse_conversation
        else:
            raise ValueError(f"Unknown conversation_parser: {conversation_parser}")

    def _auto_parse_conversation(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Auto-detect conversation format and parse accordingly."""
        # Try Qwen first (most common)
        sys, usr = parse_qwen_conversation(text)
        if sys is not None or usr is not None:
            return (sys, usr)
        # Try DeepSeek
        sys, usr = parse_deepseek_conversation(text)
        if sys is not None or usr is not None:
            return (sys, usr)
        # Try Llama3
        sys, usr = parse_llama3_conversation(text)
        return (sys, usr)

    def make_step_rewards(self, logits: torch.Tensor, token_masks: torch.Tensor) -> List[List[float]]:
        """Convert logits to step rewards using softmax probabilities.
        
        Args:
            logits: Logits tensor [batch_size, seq_len, num_labels]
            token_masks: Token mask tensor [batch_size, seq_len]
            
        Returns:
            List of lists containing positive probabilities for each sample
        """
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def process_request(self, messages: List[Dict], max_retries: int = 10) -> List[List[float]]:
        """Process HTTP request for segment rewards with retry mechanism.
        
        Args:
            messages: List of message dictionaries for the API request
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of step rewards from the API response
            
        Raises:
            Exception: If all retry attempts fail
        """
        if self.endpoint is None:
            raise ValueError("endpoint must be provided for segment reward computation")
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.endpoint,
                    json={"messages": messages},
                    proxies=self.proxies,
                    timeout=60000
                )
                step_reward = response.json()['step_rewards']
                return step_reward
            except Exception as e:
                print(f"Retry {retry_count + 1}/{max_retries}: {e}", flush=True)
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(10)
                else:
                    raise Exception(f"Failed after {max_retries} retries: {e}")

    def _default_seg_comp(self, l: int, dat: List) -> List[torch.Tensor]:
        """Default segment reward computation using external API.
        
        Args:
            l: Length of the sequence
            dat: List of tuples containing (segment_slice_dict, sys, usr)
            
        Returns:
            List of segment reward tensors
        """
        msgs = []
        for segment_slice, sys, usr in dat:
            if sys is None:
                message = [
                    {"role": "user", "content": usr},
                    {"role": "assistant", "content": [value for value in segment_slice.values()]}
                ]
            else:
                message = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                    {"role": "assistant", "content": [value for value in segment_slice.values()]}
                ]
            msgs.append(message)
        step_rewards = self.process_request(msgs)
        segment_rewards = []
        for step_reward, segment_slice in zip(step_rewards, dat):
            segment_reward = torch.zeros(l)
            for reward, rng in zip(step_reward, segment_slice[0].keys()):
                segment_reward[rng[0]:rng[1]] = reward
            segment_rewards.append(segment_reward)
        return segment_rewards

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
            if outLength > maxLength//2:
                rw -= (outLength)/maxLength
            return rw
        predicted_answer = matches[-1]
        
        if compare_latex_expressions(predicted_answer, ground_truth):
            rw = 1.0
            if outLength > maxLength//2:
                rw -= (outLength)/maxLength
            return rw
        else:
            rw = -1.0
            if outLength > maxLength//2:
                rw -= (outLength)/maxLength
            return rw

    def discount_reward(self, reward: torch.Tensor, iloc: int) -> torch.Tensor:
        """Apply discount to reward based on token position.
        
        Args:
            reward: Reward tensor
            iloc: Token position in sequence
            
        Returns:
            Discounted reward
        """
        return reward

    def compute_for_rollout(self, batch: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute RFLUX rewards; advantages are handled by adv estimator."""
        responses = batch.batch["responses"]
        response_mask = batch.batch.get("response_mask", torch.ones_like(responses, dtype=torch.bool))
        entropys = batch.batch.get(
            "entropys", torch.zeros_like(responses, dtype=torch.float32)
        )
        input_ids = batch.batch.get("input_ids", batch.batch["prompts"])

        ground_truth = []
        for i in range(len(batch)):
            data_item = batch[i]
            gt = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            if isinstance(gt, (list, tuple)):
                gt = gt[0] if len(gt) > 0 else ""
            ground_truth.append(str(gt))

        per_sample_info: List[Dict[str, Any]] = []
        data = []
        rollout_args = []
        for i in range(len(responses)):
            sample = responses[i]
            sysusr = input_ids[i]
            stopidx = sum(response_mask[i])
            outLength = stopidx
            sys, usr = self.parse_conversation(self.tokenizer.decode(sysusr))
            rollout_args.append((ground_truth[i], self.tokenizer.decode(sample[:outLength - 1]), outLength))
            segments = entropy_segmentation(entropys[i], 0, outLength, self.max_branches, self.min_gap)
            segment_rewards_dict = {(s, e): self.tokenizer.decode(sample[s:e]) for s, e in segments}
            data.append((len(entropys[i]), [(segment_rewards_dict, sys, usr)]))
            per_sample_info.append({"T": len(entropys[i])})

        with mp.Pool(mp.cpu_count()) as pool:
            seqRRs = pool.starmap(self._default_rollout_comp, rollout_args)
            segRs = pool.starmap(self._default_seg_comp, data)

        segment_rewards = []
        for i in range(len(per_sample_info)):
            seg_tensor = torch.tensor(
                [self.discount_reward(reward, iloc) for reward, iloc in zip(segRs[i][0], range(len(segRs[i][0])))],
                dtype=torch.float32,
            )
            segment_rewards.append(seg_tensor)

        rewards_tensor = torch.stack(segment_rewards)
        sequence_rewards = torch.tensor(seqRRs, dtype=torch.float32)

        if return_dict:
            return {
                "reward_tensor": rewards_tensor,
                "reward_extra_info": {"sequence_rewards": sequence_rewards},
            }
        return rewards_tensor

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute RFLUX rewards; advantages are computed separately by adv estimator."""
        return self.compute_for_rollout(batch=data, return_dict=return_dict)
