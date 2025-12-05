import torch
from collections import defaultdict
from typing import Any, Callable, List, Dict, Optional

from verl.protocol import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager

RawRewardFn = Callable[..., List[float]]

class CustomRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[RawRewardFn] = None,
        reward_fn_key: str = "data_source",
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    @staticmethod
    def entropy_segmentation(entropies: torch.Tensor, max_branches: int = 5, min_gap: int = 10):
        n = entropies.numel()
        if n <= 1 or max_branches < 1:
            return [(0, n)]
        _, idx = torch.topk(entropies, k=min(max_branches, n))
        anchors = torch.sort(idx).values.tolist()
        filtered = []
        last = -min_gap
        for a in anchors:
            if a - last >= min_gap and min_gap <= a <= n - min_gap:
                filtered.append(a)
                last = a
        cuts = []
        last_cut = 0
        for a in filtered:
            if a - last_cut >= min_gap:
                cuts.append((a, a + 1))
                last_cut = a
        returncut = []
        for i in range(len(cuts)-1):
            returncut.append((cuts[i][0], cuts[i+1][0]))
        returncut.append((last_cut, len(entropies)))
        return returncut

    def verify(self, data: DataProto):
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        logits = data.batch.get("logits", None)

        prompt_ids = data.batch.get("prompts", None)
        prompt_len = prompt_ids.shape[-1] if prompt_ids is not None else 0
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i].item()
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, [None for _ in range(len(data))])

        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        base_extra_infos = data.non_tensor_batch.get("extra_info", [dict() for _ in range(len(data))])
        entropys_tensor = data.batch.get("entropys", None)
        old_log_probs = data.batch.get("old_log_probs", None)

        results = []
        for i in range(len(data)):
            ei = dict(base_extra_infos[i]) if i < len(base_extra_infos) else {}
            ei["rollout_reward_scores"] = rollout_reward_scores[i]
            if entropys_tensor is not None:
                ei["token_entropys"] = entropys_tensor[i]
            if old_log_probs is not None:
                ei["token_log_probs"] = old_log_probs[i]
                try:
                    ei["token_probs"] = torch.exp(old_log_probs[i])
                except Exception:
                    pass

            # === segmentation logic ===
            segment_dict = {}
            if entropys_tensor is not None:
                cuts = self.entropy_segmentation(entropys_tensor[i])
                tokens = response_ids[i]
                seg_logits = logits[i] if logits is not None else None

                for (start, end) in cuts:
                    token_seg = tokens[start:end]
                    seq_seg = self.tokenizer.decode(token_seg, skip_special_tokens=True)
                    logits_seg = seg_logits[start:end] if seg_logits is not None else None

                    segment_dict[(start, end)] = {
                        "logits_output": logits_seg,
                        "token_output": token_seg,
                        "segment_sequence": seq_seg,
                    }

            ei["segments"] = segment_dict
            results.append((data_sources[i], responses_str[i], ground_truths[i], ei))

        # === compute reward ===
        scores = []
        scores = self.compute_score(
            data_sources=[r[0] for r in results],
            solution_strs=[r[1] for r in results],
            ground_truths=[r[2] for r in results],
            extra_infos=[r[3] for r in results],
            **self.reward_kwargs,
        )
        return scores, responses_str

    def compute_seq_score(self, responses_str):
        """
        输入: responses_str = list[str]
        输出: list[float]
        """
        seq_scores = []
        for seq in responses_str:
            with torch.no_grad():
                seq_scores.append(float(self.reward_model(seq)))
        return seq_scores
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        scores, response_string = self.verify(data)
        seq_scores = self.compute_seq_score(response_string)

        prompt_ids = data.batch.get("prompts", None)
        prompt_len = prompt_ids.shape[-1] if prompt_ids is not None else 0
        valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

        returnvalue = []
        for i in range(len(data)):
            score = scores[i]
            seqscore = seq_scores[i]
            returnvalue.append({"seg":score, "seq":seqscore})
        return returnvalue