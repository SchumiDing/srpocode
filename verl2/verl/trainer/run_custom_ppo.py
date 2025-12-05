"""
Run a small demo that wires the custom actor, reward manager and trainer.

This script:
- Provides a small MockEngine that returns DataProto with responses, old_log_probs and entropys.
- Provides a tiny MockTokenizer used by CustomRewardManager for decoding.
- Defines a simple batch-style reward_fn that consumes token_entropys.
- Instantiates CustomPPOTrainer (from verl/trainer/ppo/custom_ray_trainer.py),
  sets simple policy and critic, runs one rollout_and_train_step and prints stats.

Prerequisite:
- Place the previously provided files under:
    verl/workers/actor/custom_actor.py
    verl/workers/reward_manager/custom_reward_manager.py
    verl/trainer/ppo/custom_ray_trainer.py
  or ensure they are importable in PYTHONPATH.

Run:
    python examples/run_custom_ppo.py
"""

import torch
import torch.nn as nn
from torch import tensor
from verl.protocol import DataProto

# Import the custom components (ensure these files are on PYTHONPATH)
from verl.workers.actor.custom_actor import CustomActor
from verl.workers.reward_manager.custom_reward_manager import CustomRewardManager
from verl.trainer.ppo.custom_ray_trainer import CustomPPOTrainer

# -------------------------
# Mock / Helper Components
# -------------------------
class MockTokenizer:
    """Very small tokenizer mock used for decode in reward manager."""
    def decode(self, token_ids, skip_special_tokens=True):
        # token_ids: 1D tensor or list of token ids
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join([str(x) for x in token_ids])

class MockEngine:
    """A mock engine that returns a DataProto with necessary keys.
    generate_sequences(prompts: DataProto, **kwargs) -> DataProto
    """
    def __init__(self, vocab_size=1000, response_length=8, prompt_length=4):
        self.vocab_size = vocab_size
        self.response_length = response_length
        self.prompt_length = prompt_length

    def generate_sequences(self, prompts: DataProto, **kwargs):
        # Determine batch size from provided prompts (if present), else use 2
        if prompts is not None and "prompts" in prompts.batch:
            prompts_tensor = prompts.batch["prompts"]
            bsz = prompts_tensor.shape[0]
            prompt_len = prompts_tensor.shape[-1]
        else:
            bsz = 2
            prompt_len = self.prompt_length

        # Create fake prompt ids (copy incoming if present)
        if prompts is not None and "prompts" in prompts.batch:
            prompt_ids = prompts.batch["prompts"]
        else:
            prompt_ids = torch.randint(0, self.vocab_size, (bsz, prompt_len), dtype=torch.long)

        # Generate fake response ids
        responses = torch.randint(0, self.vocab_size, (bsz, self.response_length), dtype=torch.long)
        # Create input_ids = concat(prompt + response)
        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        seq_len = input_ids.shape[-1]

        # attention mask: 1 for all tokens here (no padding)
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)
        # position ids: simple range per sample
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

        # Fake old_log_probs: negative numbers (log-probs)
        # shape (bsz, response_length)
        old_log_probs = -torch.rand(bsz, self.response_length)

        # Fake entropys: positive floats
        entropys = torch.rand(bsz, self.response_length) * 2.0  # entropy in bits/nats

        # Build batch dict expected by rest of pipeline
        batch = {
            "prompts": prompt_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # reward manager expects "old_log_probs" and "entropys" to be present
            "old_log_probs": old_log_probs,
            "entropys": entropys,
        }

        # Return DataProto (on CPU)
        return DataProto.from_dict(tensors=batch)

class SimpleCritic:
    """A simple critic that returns zero values matching responses shape."""
    def __init__(self):
        # underlying small module to allow optimizer attachment
        self.model = nn.Linear(4, 4)  # dummy

    def __call__(self, data_proto: DataProto):
        # Return zeros shaped like (bsz, response_length)
        responses = data_proto.batch["responses"]
        bsz, resp_len = responses.shape
        return torch.zeros((bsz, resp_len), dtype=torch.float32)

# -------------------------
# Reward function example
# -------------------------
def entropy_based_reward_batch(data_sources, solution_strs, ground_truths, extra_infos=None, **kwargs):
    """Batch-style reward function.

    Score = 1.0 - (avg_token_entropy / scale), clipped into [-1.0, 1.0].
    Returns list[dict] with 'score' and 'avg_entropy' for extra_info collection.
    """
    extra_infos = extra_infos or [None] * len(solution_strs)
    results = []
    for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos):
        ent = None
        if ei is not None:
            ent = ei.get("token_entropys", None)
        if ent is not None:
            # ent can be a torch tensor; convert to scalar avg
            if isinstance(ent, torch.Tensor):
                avg_ent = float(ent.mean().item())
            else:
                # if list/np array
                avg_ent = float(torch.tensor(ent).mean().item())
            score = 1.0 - (avg_ent / 5.0)
        else:
            avg_ent = None
            score = 0.0
        score = max(-1.0, min(1.0, score))
        results.append({"score": float(score), "avg_entropy": avg_ent})
    return results

# -------------------------
# Demo main
# -------------------------
def main():
    torch.manual_seed(42)

    # Create mock tokenizer and engine
    tokenizer = MockTokenizer()
    mock_engine = MockEngine(vocab_size=5000, response_length=8, prompt_length=4)

    # Instantiate the CustomActor with the MockEngine
    actor = CustomActor(engine=mock_engine, config={"response_length": 8, "prompt_length": 4, "temperature": 1.0, "calculate_entropy": True})

    # Make sure CustomRewardManager will be able to use the tokenizer and reward fn
    reward_manager = CustomRewardManager(tokenizer=tokenizer, num_examine=2, compute_score=entropy_based_reward_batch, reward_fn_key="data_source")

    # Build trainer using actor_engine=mock_engine (CustomPPOTrainer internally instantiates its own CustomActor in the original)
    # Here we pass the mock_engine to CustomPPOTrainer so it creates an actor consistent with it.
    trainer = CustomPPOTrainer(actor_engine=mock_engine, tokenizer=tokenizer, reward_fn=entropy_based_reward_batch, reward_fn_key="data_source", num_examine=2)

    # Create simple policy and critic modules and optimizer so trainer.set_policy_and_optim works
    policy = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
    critic_model = SimpleCritic()
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(critic_model.model.parameters()), lr=1e-4)

    trainer.set_policy_and_optim(policy_module=policy, critic_module=critic_model, optimizer=optimizer)

    # Build prompts DataProto - small batch of 2 prompts
    bsz = 2
    prompt_len = 4
    prompts_tensor = torch.randint(0, 1000, (bsz, prompt_len), dtype=torch.long)
    # minimal attention_mask/position_ids required by some actor implementations
    attention_mask = torch.ones((bsz, prompt_len), dtype=torch.long)
    position_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)

    prompts_dp = DataProto.from_dict(tensors={
        "prompts": prompts_tensor,
        "input_ids": prompts_tensor,           # for this demo, input_ids == prompts
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })

    # Run one rollout + train step
    out = trainer.rollout_and_train_step(prompts_dp)

    print("Run stats:")
    for k, v in out["stats"].items():
        print(f"  {k}: {v}")

    # Print reward extra info if any
    print("\nReward extra info (sample):")
    print(out.get("reward_extra", {}))

    # Print advantages shapes
    adv = out["advantages"]
    print("\nAdvantages shape:", adv.shape)
    print("Returns shape:", out["returns"].shape)

if __name__ == "__main__":
    main()