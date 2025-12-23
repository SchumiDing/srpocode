
from typing import Callable, List, Optional, Tuple, Dict, Any
import logging
import re
import torch
import torch.nn.functional as F
from verl.workers.reward_manager import register
import requests
import time
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

endpoint = "http://10.102.248.155:4997/v1/step_rewards"
logger = logging.getLogger(__name__)
proxies = {
    "http": None,
    "https": None
}
from openai import OpenAI
import json
import multiprocessing as mp
from sympy import simplify, parse_expr
from sympy.parsing.latex import parse_latex
client = OpenAI(api_key="sk-ce18ce62795d4af6866c9af5ebac02d4", base_url="https://api.deepseek.com")
def if_same(y_hat, y):
    prompt = "Please judge if the following two mathematical expression refers to the same result, and please ignore syntax errors:"+\
        f"y_hat: {y_hat}"+ "\n" +\
        f"y: {y}"+ "\n" +\
        "Please only return a json with key 'answer' and value True or False\n **Notice** Make sure you only return a json, no other text or comments."
    answer = None
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            resp = response.choices[0].message.content.strip()
            resp = resp.replace("```json", "").replace("```", "")
            answer = json.loads(resp)['answer']
            break
        except:
            print(resp, flush=True)
            print("Error in if_same", flush=True)
            continue
    return answer


def entropy_segmentation(entropies: torch.Tensor, start_idx: int, outLength: int, max_branches: int = 5, min_gap: int = 10) -> List[Tuple[int, int]]:
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
    if outLength-start_idx < max_branches+1:
        return [(start_idx, outLength)]
    # 从 start_idx 开始，找到 max_branches 个 anchor， 去除间距小于 min_gap 的 anchor
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
    # 将每个 anchor 视为 cut 点（点到点之间形成 segments）
    for a in topk_idx:
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
            segments.append((prev, c))
            prev = c
        # 最后一段从最后 anchor 到结尾
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
            # 在 s 与 cur 之间存在间隙，填补
            sanitized.append((cur, s))
        sanitized.append((s, e))
        cur = e
    if cur < n:
        sanitized.append((cur, n))

    if len(sanitized) == 0:
        return [(0, n)]
    return sanitized
def parse_llama3_conversation(text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = r'<\|start_header_id\|>([^<]+)<\|end_header_id\|>\s*\n{0,2}(.*?)\s*<\|eot_id\|>'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    system_content = None
    user_content = None
    
    for role, content in matches:
        role = role.strip()
        content = content.strip()
        
        if role == "system" and system_content is None:
            system_content = content
        elif role == "user" and user_content is None:
            user_content = content
    
    return (system_content, user_content)
def parse_qwen_conversation(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析Qwen聊天模板应用后的文本，提取system内容和user内容
    
    Args:
        text: 应用了Qwen chat template后的文本
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (system_content, user_content)
    """
    system_content = None
    user_content = None
    
    # 匹配system消息
    system_pattern = r'<\|im_start\|>system\s*\n(.*?)<\|im_end\|>'
    system_matches = re.findall(system_pattern, text, re.DOTALL)
    
    if system_matches:
        # 取第一个system消息
        system_content = system_matches[0].strip()
    
    # 匹配user消息
    user_pattern = r'<\|im_start\|>user\s*\n(.*?)<\|im_end\|>'
    user_matches = re.findall(user_pattern, text, re.DOTALL)
    
    if user_matches:
        # 取第一个user消息
        user_content = user_matches[0].strip()
        
        # # 清理可能的tool_response内容
        # if '<tool_response>' in user_content:
        #     # 提取tool_response之前的内容作为真正的user content
        #     parts = user_content.split('<tool_response>')
        #     if parts[0].strip():
        #         user_content = parts[0].strip()
    
    return (system_content, user_content)

def parse_deepseek_conversation(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析DeepSeek聊天模板应用后的文本，提取system内容和user内容
    
    DeepSeek chat template格式:
    - System prompt: 直接跟在bos_token后面，在第一个<｜User｜>之前
    - User消息: <｜User｜> + content
    - Assistant消息: <｜Assistant｜> + content + <｜end▁of▁sentence｜>
    
    Args:
        text: 应用了DeepSeek chat template后的文本
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (system_content, user_content)
    """
    system_content = None
    user_content = None
    
    # DeepSeek使用全角竖线 ｜ (U+FF5C) 和特殊下划线 ▁ (U+2581)
    # 提取system content: 在第一个<｜User｜>之前的内容（去掉可能的bos_token）
    user_tag = '<｜User｜>'
    assistant_tag = '<｜Assistant｜>'
    
    # 找到第一个User标记的位置
    first_user_pos = text.find(user_tag)
    
    if first_user_pos > 0:
        # system content 是 bos_token 之后到第一个 <｜User｜> 之前的内容
        system_content = text[:first_user_pos].strip()
        # 去掉可能的bos_token (如 <｜begin▁of▁sentence｜>)
        bos_pattern = r'^<｜begin▁of▁sentence｜>'
        system_content = re.sub(bos_pattern, '', system_content).strip()
        if not system_content:
            system_content = None
    
    # 提取user content: <｜User｜> 到下一个标记之间的内容
    if first_user_pos != -1:
        # 从第一个User标记之后开始
        after_user_tag = first_user_pos + len(user_tag)
        remaining_text = text[after_user_tag:]
        
        # 找到下一个标记的位置（可能是Assistant、tool相关标记等）
        # 可能的结束标记
        end_markers = [
            '<｜Assistant｜>',
            '<｜tool▁calls▁begin｜>',
            '<｜tool▁outputs▁begin｜>',
            '<｜end▁of▁sentence｜>',
            '<｜User｜>',  # 如果有多轮对话
        ]
        
        # 找到最近的结束标记
        end_pos = len(remaining_text)
        for marker in end_markers:
            pos = remaining_text.find(marker)
            if pos != -1 and pos < end_pos:
                end_pos = pos
        
        user_content = remaining_text[:end_pos].strip()
        if not user_content:
            user_content = None
    
    return (system_content, user_content)


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
@register("pure")
class PureRewardManager:
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
        
        self.alpha = 1
        self.max_branches = int(max_branches)
        self.min_gap = int(min_gap)
        self.seg_comp_fn = seg_comp_fn
        self.rollout_comp_fn = rollout_comp_fn
        self.tokenizer = tokenizer
        self.scv=(torch.sqrt(torch.tensor(1/24)))
        self.lamb = 0.9
        self.temperature = 0.1
    def make_step_rewards(self,logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res
    def process_request(self, messages, max_retries=10):
        """处理推理请求"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.post(endpoint, json={"messages": messages}, proxies=proxies, timeout=60000)
                step_reward = response.json()['step_rewards']
                return step_reward
            except Exception as e:
                print(f"Retry {retry_count + 1}/{max_retries}: {e}", flush=True)
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(10)
                else:
                    raise Exception(f"Failed after {max_retries} retries: {e}")
        
    def _default_seg_comp(self, l, dat) -> float:
        """
        默认 segment reward：使用 -mean(log_prob) 作为示例（log_prob 越高，-mean 越低）
        如果输入为空或标量，返回 0.0。
        segment_slice: 1D tensor, 每个 time step 的 log_prob proxy（可能是负 entropy，也可能是 log_prob）
        """
        msgs = []
        for segment_slice, sys, usr in dat:
            if sys is None :
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
        sum_ = sum([torch.exp(-x/self.temperature) for x in step_rewards])
        for i in range(len(step_rewards)):
            step_rewards[i] = torch.exp(-step_rewards[i]/self.temperature) / sum_ * step_rewards[i]
        minLoc = torch.argmin(step_rewards)
        minV = step_rewards[minLoc]
        for i in range(len(step_rewards)):
            if i < minLoc:
                step_rewards[i] = minV
            else:
                step_rewards[i] = 0
        segment_rewards = []
        for step_reward, segment_slice in zip(step_rewards, dat):
            segment_reward = torch.zeros(l) 
            for reward, rng in zip(step_reward, segment_slice[0].keys()):
                segment_reward[rng[1]-1] = reward
            segment_rewards.append(segment_reward)
        return segment_rewards
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
        # if len(sequence) > 1000:
        #     value -= len(sequence)/1000
        
    def discount_reward(self, reward: torch.Tensor, iloc: int):
        """
        该函数用于对较长的token进行折扣，iloc为token在序列中的位置
        """
        return reward
        # baseline = 4096
        # discount = reward.clone()
        # if iloc > baseline//2 and iloc <= baseline*3//4:
        #     discount -= 0.25
        # elif iloc > baseline*3//4:
        #     discount -= 0.5
        # return discount

    def compute_for_rollout(self, input_ids,ground_truth, response_mask, rollout_outputs: List[str], entropys: torch.Tensor, repeat_times: int = 1) -> Dict[str, Any]:
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
        # for i in range(len(rollout_outputs)):
        #     print(
        #         "Response", self.tokenizer.decode(rollout_outputs[i]), flush=True
        #     )
        data = []
        seqRRs = []
        # print(entropys[0], flush=True)
        # print(entropys[0][-outLength:], flush=True)
        outl = 0
        
        rollout_args = []
        for i in range(len(rollout_outputs)):
            sample = rollout_outputs[i]
            sysusr = input_ids[i]
            stopidx = sum(response_mask[i])
            outLength = stopidx
            outl += outLength
            start_idx = 0
            # print("input", self.tokenizer.decode(sysusr), flush=True)
            # sys, usr = parse_llama3_conversation(self.tokenizer.decode(sysusr))
            # sys, usr = parse_qwen_conversation(self.tokenizer.decode(sysusr))
            sys, usr = parse_deepseek_conversation(self.tokenizer.decode(sysusr))
            # print("sys", sys, flush=True)
            # print("usr", usr, flush=True)
            rollout_args.append((ground_truth[i], self.tokenizer.decode(sample[:outLength-1]), outLength))
            segments = entropy_segmentation(entropys[i], start_idx, outLength)
            segment_rewards_dict = {(s, e): self.tokenizer.decode(sample[s:e]) for s, e in segments}
            data.append((len(entropys[i]),[(segment_rewards_dict, sys, usr)]))
            per_sample_info.append({
                'segments': segments,
                'T': len(entropys[i]),
                "l": sum(response_mask[i])
                })
        print(f"Start computing rollout rewards", flush=True)
        with mp.Pool(mp.cpu_count()) as pool:
            print(f"Start computing sequence rewards", flush=True)
            seqRRs = pool.starmap(self._default_rollout_comp, rollout_args)
            print(f"End computing sequence rewards", flush=True)
            print(f"Start computing segment rewards", flush=True)
            segRs = pool.starmap(self._default_seg_comp, data)
            print(f"End computing segment rewards", flush=True)
        print(f"End computing rollout rewards", flush=True)
        for i, seqRReward in enumerate(seqRRs):
            per_sample_info[i]['seq_reward'] = seqRReward
        print(f"End computing rollout rewards", flush=True)
        
        
        print(f"outl: {outl}", flush=True)
        # print(len(segRs), flush=True)
        print(len(per_sample_info), flush=True)
        for i in range(len(per_sample_info)):
            segRs[i][0] = torch.tensor([self.discount_reward(reward, iloc) for reward, iloc in zip(segRs[i][0], range(len(segRs[i][0])))])
            for j in range(len(segRs[i][0])):
                sum_ = torch.zeros((), dtype=segRs[i][0].dtype, device=segRs[i][0].device)
                for k in range(j, len(segRs[i][0])):
                    # 使用 float 幂避免 torch.pow 对纯 float 抛出类型错误
                    weight = self.lamb ** (k - j)
                    sum_ += weight * segRs[i][0][k]
                segRs[i][0][j] = sum_
            per_sample_info[i]['segment_rewards'] = torch.tensor(segRs[i][0])

        for i in range(0, len(per_sample_info), repeat_times):
            avg = 0
            num = 0
            for j in range(i, i+repeat_times):
                avg += torch.sum(per_sample_info[j]['segment_rewards'])
                num += len(per_sample_info[j]['segment_rewards'])
            avg /= num
            per_sample_info[i]['per_timestep_adv'] = per_sample_info[i]['segment_rewards'] - avg
        
        rollout_raw_reward = [info['seq_reward'] for info in per_sample_info]
        rewards_tensor = [info['seq_reward'] for info in per_sample_info]
        for group_idx in range(len(rollout_raw_reward)//repeat_times):
            rollout_reward = rollout_raw_reward[(group_idx)*repeat_times:(group_idx+1)*repeat_times]
            tz_scores = []
            for idx in range(len(rollout_reward)):
                mean=0
                for j in range(len(rollout_reward)):
                    if idx != j:
                        mean += rollout_reward[j]
                mean /= len(rollout_reward) - 1
                tz_scores.append(rollout_reward[idx] - mean)
            for j in range(group_idx*repeat_times, (group_idx+1)*repeat_times):
                    per_sample_info[j]['beta'] = tz_scores[j-group_idx*repeat_times]

        for idx, info in enumerate(per_sample_info):
            beta = info['beta']
            per_timestep_adv = beta + torch.zeros(info['T'])
            info['per_timestep_adv'] = per_timestep_adv
        i = torch.randint(0, len(per_sample_info), (1,)).item()
        # rewards_tensor = None
        # rewards_tensor = torch.tensor([info['per_timestep_adv'] for info in per_sample_info])
        
        # print(f"{i}th Avg Adv: ", mean(per_sample_info[i]['per_timestep_adv']), flush=True)
        print(f"{i}th Seg Rewards: ", torch.mean(per_sample_info[i]['segment_rewards']), flush=True)
        print(f"{i}th Beta: ", per_sample_info[i]['beta'], flush=True)
        print(f"{i}th seq Reward: ", per_sample_info[i]['seq_reward'], flush=True)

        betaTensor = torch.tensor([info["seq_reward"] for info in per_sample_info], dtype=torch.float16)
        advTensor = torch.stack([info['per_timestep_adv'] for info in per_sample_info])
        return rewards_tensor, betaTensor, advTensor
    
    # def __call__(self, data: DataProto, config: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    #     return self.compute_for_rollout(data.batch)