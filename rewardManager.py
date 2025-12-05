
from typing import Callable, List, Optional, Tuple, Dict, Any
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def entropy_segmentation(entropies: torch.Tensor, max_branches: int = 5, min_gap: int = 10) -> List[Tuple[int, int]]:
    """
    根据 entropies 对序列分段，返回一组 (start, end) 区间，end 不包含（闭开区间 [start, end)）。
    原始逻辑基础上修正边界问题并保证至少返回一个 segment 覆盖整个序列。

    Args:
        entropies: 1D tensor, length T
        max_branches: 最大候选 anchors 数量（选择 top-k 高 entropy 位点作为 anchor）
        min_gap: 两个 anchor 之间最小距离，同时要求 anchor 离序列边缘至少 min_gap，用以避免过短 segment

    Returns:
        List[(start, end)] covering [0, T)
    """
    if not isinstance(entropies, torch.Tensor):
        entropies = torch.tensor(entropies)
    n = int(entropies.numel())
    if n <= 1 or max_branches < 1:
        return [(0, n)]
    k = min(max_branches, n)
    # topk 返回的是值与索引，取索引作为 anchors
    _, idx = torch.topk(entropies, k=k)
    anchors = torch.sort(idx).values.tolist()
    filtered = []
    last = -min_gap
    # 过滤：保证 anchors 之间以及与边缘的距离
    for a in anchors:
        if (a - last) >= min_gap and (min_gap <= a <= n - min_gap):
            filtered.append(a)
            last = a
    cuts = []
    last_cut = 0
    # 将每个 anchor 视为 cut 点（点到点之间形成 segments）
    for a in filtered:
        if a - last_cut >= min_gap:
            # 为保证 segment 能分割出区间，cut 记录 anchor 本身
            cuts.append(a)
            last_cut = a
    # 构建 segments，由于 cuts 是 anchor positions (单点)，构建区间 [prev_cut, curr_cut)
    segments: List[Tuple[int, int]] = []
    if len(cuts) == 0:
        segments = [(0, n)]
    else:
        prev = 0
        for c in cuts:
            # 保证边界合法
            s = max(0, min(prev, n))
            e = max(s, min(c, n))
            if e > s:
                segments.append((s, e))
            prev = c
        # 最后一段从最后 anchor 到结尾
        if prev < n:
            segments.append((prev, n))
    # 最后再做一次 sanitize：合并长度为0的 segment 并确保覆盖 [0,n)
    sanitized: List[Tuple[int, int]] = []
    cur = 0
    for (s, e) in segments:
        s = max(0, min(n, int(s)))
        e = max(0, min(n, int(e)))
        if e <= s:
            continue
        if s > cur:
            # 在 s 与 cur 之间存在间隙，填补
            sanitized.append((cur, s))
        sanitized.append((s, e))
        cur = e
    if cur < n:
        sanitized.append((cur, n))
    # 如果 sanitized 为空（极端情况），返回整体
    if len(sanitized) == 0:
        return [(0, n)]
    return sanitized


class NaiveRewardManager:
    def __init__(self,
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

    @staticmethod
    def _default_seg_comp(segment_slice: torch.Tensor, sample_meta: Dict[str, Any]) -> float:
        """
        默认 segment reward：使用 -mean(log_prob) 作为示例（log_prob 越高，-mean 越低）
        如果输入为空或标量，返回 0.0。
        segment_slice: 1D tensor, 每个 time step 的 log_prob proxy（可能是负 entropy，也可能是 log_prob）
        """
        if not isinstance(segment_slice, torch.Tensor) or segment_slice.numel() == 0:
            return 0.0
        return float(-segment_slice.mean().item())

    @staticmethod
    def _default_rollout_comp(segment_rewards: List[float], sample_meta: Dict[str, Any]) -> float:
        """
        默认 rollout 级别组合 reward：直接求和
        """
        if not segment_rewards:
            return 0.0
        return float(sum(segment_rewards))

    @staticmethod
    def _extract_step_proxy(sample: Dict[str, Any]) -> torch.Tensor:
        """
        提取每个时间步的 proxy 值用于 segmentation 与 seg_comp：
        优先级：
        1) 'entropy' 字段（T,)
        2) 'log_probs' 或 'action_log_probs'（T,)
        3) 'logits'（T, A） -> 计算每步的离散熵
        4) 'action_dist' 提供 distribution 的情况下尝试调用其 entropy()（兼容性）
        5) fallback 全 0 Tensor

        返回 1D tensor 长度 T (T 可为0)
        """
        # 1: entropy
        if 'entropy' in sample and isinstance(sample['entropy'], torch.Tensor):
            ent = sample['entropy']
            if ent.dim() == 1:
                return ent.clone().detach()
            # otherwise try to reduce
            return ent.view(-1).clone().detach()

        # 2: log_probs
        for k in ('log_probs', 'action_log_probs', 'action_logprob'):
            if k in sample and isinstance(sample[k], torch.Tensor):
                lp = sample[k]
                if lp.dim() == 1:
                    # convert log_prob -> abs(log_prob) as proxy (higher magnitude -> higher "surprise")
                    return lp.clone().detach().abs()
                else:
                    return lp.view(-1).clone().detach().abs()

        # 3: logits -> compute discrete entropy per timestep
        if 'logits' in sample and isinstance(sample['logits'], torch.Tensor):
            logits = sample['logits']
            # logits shape could be (T, A) or (A,) (single step)
            if logits.dim() == 2:
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # (T,)
                return ent.detach()
            elif logits.dim() == 1:
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * (probs + 1e-12).log()).sum()
                return torch.tensor([float(ent.detach())])
            else:
                # unsupported shape
                pass

        # 4: action_dist object with entropy method
        if 'action_dist' in sample:
            ad = sample['action_dist']
            try:
                ent = ad.entropy()
                if isinstance(ent, torch.Tensor):
                    return ent.view(-1).clone().detach()
            except Exception:
                pass

        # fallback: try to infer length from 'actions' or 'length'
        T = None
        if 'length' in sample:
            T = int(sample['length'])
        elif 'actions' in sample and isinstance(sample['actions'], torch.Tensor):
            T = int(sample['actions'].shape[0])
        elif 'log_probs' in sample and isinstance(sample['log_probs'], torch.Tensor):
            T = int(sample['log_probs'].shape[0])
        if T is None:
            return torch.tensor([])
        return torch.zeros(T)

    def compute_for_rollout(self, rollout_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        主入口：对 rollout 中每个样本做 segmentation、seg reward、rollout reward、归一化与 final adv 计算。
        Args:
            rollout_outputs: list of sample dicts (任意结构，但常见包含 'log_probs'/'logits'/'actions'/'length' 等)
        Returns:
            dict with key 'per_sample' -> list of per-sample dicts (见文件头说明)
        """
        per_sample_info: List[Dict[str, Any]] = []
        if rollout_outputs is None:
            return {'per_sample': per_sample_info}

        # 第一遍：计算 segment 划分与原始 seg reward、rollout raw reward
        rollout_raw_rewards: List[float] = []
        for sample in rollout_outputs:
            # 提取 step-level proxy（entropy 或 abs(log_prob)）
            step_proxy = self._extract_step_proxy(sample)  # 1D tensor (T,)
            T = int(step_proxy.numel())

            try:
                segments = entropy_segmentation(step_proxy, max_branches=self.max_branches, min_gap=self.min_gap)
            except Exception as e:
                logger.exception("Segmentation failed, fallback to single segment: %s", e)
                segments = [(0, T)]

            segment_rewards: List[float] = []
            for (s, e) in segments:
                if s >= e or s < 0 or e > T:
                    seg_slice = torch.tensor([])
                else:
                    # prepare a slice for seg_comp_fn: we pass the same proxy as "log_probs" like vector
                    seg_slice = step_proxy[s:e].clone().detach()
                if self.seg_comp_fn is not None:
                    try:
                        sr = float(self.seg_comp_fn(seg_slice, sample))
                    except Exception:
                        logger.exception("Custom seg_comp_fn failed, using default.")
                        sr = float(self._default_seg_comp(seg_slice, sample))
                else:
                    sr = float(self._default_seg_comp(seg_slice, sample))
                segment_rewards.append(sr)

            # rollout-level comp
            if self.rollout_comp_fn is not None:
                try:
                    r_rollout = float(self.rollout_comp_fn(segment_rewards, sample))
                except Exception:
                    logger.exception("Custom rollout_comp_fn failed, using default.")
                    r_rollout = float(self._default_rollout_comp(segment_rewards, sample))
            else:
                r_rollout = float(self._default_rollout_comp(segment_rewards, sample))

            per_sample_info.append({
                'sample_meta': sample,
                'segments': segments,
                'segment_rewards': segment_rewards,
                'rollout_reward_raw': r_rollout,
                'T': T
            })
            rollout_raw_rewards.append(r_rollout)

        # 第二步：样本内归一化（segment_norms）
        for info in per_sample_info:
            seg_vals = info.get('segment_rewards', [])
            if len(seg_vals) == 0:
                info['segment_norms'] = []
            elif len(seg_vals) == 1:
                info['segment_norms'] = [0.5]
            else:
                arr = torch.tensor(seg_vals, dtype=torch.float32)
                mean_val = float(arr.mean().item())
                std_val = float(arr.std().item())
                
                if std_val == 0:
                    info['segment_norms'] = [0.5 for _ in seg_vals]
                else:
                    z_scores = (arr - mean_val) / std_val
                    info['segment_norms'] = z_scores.tolist()

        # 第三步：rollout-level 归一化 -> beta
        betas: List[float] = []
        if len(rollout_raw_rewards) == 0:
            betas = []
        elif len(rollout_raw_rewards) == 1:
            betas = [0.5]
        else:
            arr = torch.tensor(rollout_raw_rewards, dtype=torch.float32)
            mean_val = float(arr.mean().item())
            std_val = float(arr.std().item())
            
            if std_val == 0:
                betas = [0.5 for _ in rollout_raw_rewards]
            else:
                z_scores = (arr - mean_val) / std_val
                betas = z_scores.tolist()

        # 第四步：构建 final advantages 并广播到时间步
        for idx, info in enumerate(per_sample_info):
            seg_norms = info.get('segment_norms', [])
            beta = float(betas[idx]) if idx < len(betas) else 0.0
            T = int(info.get('T', 0))
            per_timestep_adv = torch.zeros(T, dtype=torch.float32) if T > 0 else torch.tensor([])
            final_advs: List[float] = []
            segments = info.get('segments', [])
            for j, normv in enumerate(seg_norms):
                final = float(normv + self.alpha * beta)
                final_advs.append(final)
                # write back to per-timestep advantage
                if j < len(segments):
                    s, e = segments[j]
                    s = max(0, min(T, int(s)))
                    e = max(0, min(T, int(e)))
                    if e > s and T > 0:
                        per_timestep_adv[s:e] = float(final)
            # store
            info['beta'] = float(beta)
            info['final_advantages'] = final_advs
            info['per_timestep_adv'] = per_timestep_adv

        return {'per_sample': per_sample_info}