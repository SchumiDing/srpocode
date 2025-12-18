# Reward Manager Algorithms Usage Guide

## Overview

This directory contains reward manager implementations that compute reward signals
for PPO-style training. Advantages are now computed by adv estimators (e.g.,
`grpo`, `srpo`, `srpo2`, `srpo3`, `srpo4`, `rflux`). All algorithms support
multiple model output formats (Qwen, Llama3, DeepSeek) and multi-node multi-GPU
inference through Ray.

## Available Algorithms

### 1. GRPO (Group Relative Policy Optimization)
- **Registration name**: `grpo`
- **Description**: Computes sequence-level rewards and performs within-group normalization to use relative advantages instead of absolute rewards.
- **Key features**:
  - Only uses sequence-level rewards
  - Normalizes rewards within groups (repeat_times samples)
  - Broadcasts normalized advantages to all timesteps
  - Adaptive minimum std threshold based on group size

### 2. SRPO (Segment-based Reward Policy Optimization)
- **Registration name**: `srpo`
- **Description**: Computes both segment-level and sequence-level rewards, then combines them with normalization.
- **Key features**:
  - Uses entropy-based segmentation
  - Computes segment-level rewards via external API
  - Combines segment and sequence rewards
  - Supports multiple conversation parsers (Qwen by default)

### 3. SRPO2
- **Registration name**: `srpo2`
- **Description**: Similar to SRPO but uses DeepSeek conversation parsing and subtracts mean from beta.
- **Key differences from SRPO**:
  - Uses `parse_deepseek_conversation` by default
  - Beta calculation: `tz_scores - mean` (centered)
  - Uses maxLength = 8192 for length penalty

### 4. SRPO3
- **Registration name**: `srpo3`
- **Description**: Similar to SRPO2 but adds mean segment reward to sequence reward.
- **Key differences from SRPO2**:
  - `rollout_raw_reward = seq_reward + mean_seqReward`
  - Uses DeepSeek conversation parsing

### 5. SRPO4
- **Registration name**: `srpo4`
- **Description**: Similar to SRPO but uses within-group std for segment normalization.
- **Key differences from SRPO**:
  - Uses within-group std for segment normalization instead of fixed scv
  - Uses Qwen conversation parsing
  - Beta calculation: `tz_scores - mean` (centered)

### 6. RFLUX
- **Registration name**: `rflux`
- **Description**: Combines sequence reward with mean segment rewards, uses only beta as advantage.
- **Key features**:
  - `rollout_raw_reward = seq_reward + mean(segment_rewards)`
  - Only uses beta (sequence-level) as advantage, no segment-level component
  - Uses DeepSeek conversation parsing
  - Uses std normalization for beta

## Configuration Parameters

When running `main_ppo.py`, you need to specify the following parameters in your config file:

### Basic Configuration

```yaml
reward_manager:
  source: "register"  # Use registered reward managers
  name: "grpo"  # Options: "grpo", "srpo", "srpo2", "srpo3", "srpo4", "rflux"
  
reward_model:
  reward_kwargs:
    # Common parameters
    alpha: 0.5  # Weight parameter (for compatibility, not used in all algorithms)
    max_branches: 5  # Maximum branches for segmentation (used in SRPO variants)
    min_gap: 10  # Minimum gap for segmentation (used in SRPO variants)
    
    # For SRPO variants that need segment rewards
    endpoint: "http://your-endpoint:port/v1/step_rewards"  # Required for SRPO variants
    proxies:  # Optional proxy configuration
      http: null
      https: null
    
    # Conversation parser selection (for SRPO variants)
    conversation_parser: "qwen"  # Options: "qwen", "llama3", "deepseek", "auto"
    # "auto" will try to detect the format automatically
```

### Example Configurations

#### GRPO Configuration
```yaml
reward_manager:
  source: "register"
  name: "grpo"

reward_model:
  reward_kwargs:
    alpha: 0.5
    max_branches: 5
    min_gap: 10
```

#### SRPO Configuration (with segment rewards)
```yaml
reward_manager:
  source: "register"
  name: "srpo"

reward_model:
  reward_kwargs:
    max_branches: 5
    min_gap: 10
    endpoint: "http://localhost:4997/v1/step_rewards"
    conversation_parser: "qwen"  # or "llama3", "deepseek", "auto"
    proxies:
      http: null
      https: null
```

#### SRPO2 Configuration (DeepSeek format)
```yaml
reward_manager:
  source: "register"
  name: "srpo2"

reward_model:
  reward_kwargs:
    max_branches: 5
    min_gap: 10
    endpoint: "http://localhost:4997/v1/step_rewards"
    conversation_parser: "deepseek"  # DeepSeek format
```

#### SRPO3 Configuration
```yaml
reward_manager:
  source: "register"
  name: "srpo3"

reward_model:
  reward_kwargs:
    max_branches: 5
    min_gap: 10
    endpoint: "http://localhost:4997/v1/step_rewards"
    conversation_parser: "deepseek"
```

#### SRPO4 Configuration (with group std normalization)
```yaml
reward_manager:
  source: "register"
  name: "srpo4"

reward_model:
  reward_kwargs:
    max_branches: 5
    min_gap: 10
    endpoint: "http://localhost:4997/v1/step_rewards"
    conversation_parser: "qwen"
```

#### RFLUX Configuration
```yaml
reward_manager:
  source: "register"
  name: "rflux"

reward_model:
  reward_kwargs:
    max_branches: 5
    min_gap: 10
    endpoint: "http://localhost:4997/v1/step_rewards"
    conversation_parser: "deepseek"
```

## Multi-Node Multi-GPU Support

All algorithms support multi-node multi-GPU inference through Ray:

1. **Ray Configuration**: The algorithms use `ThreadPoolExecutor` instead of `multiprocessing.Pool` to avoid deadlocks in Ray environments.

2. **Distributed Setup**: 
   - Ray automatically handles distributed execution
   - Each worker can compute rewards independently
   - Results are aggregated automatically

3. **Configuration Example**:
```yaml
trainer:
  nnodes: 2  # Number of nodes
  n_gpus_per_node: 8  # GPUs per node
  
ray_kwargs:
  ray_init:
    address: "auto"  # Auto-connect to existing cluster or start new one
```

## Usage in Training

The algorithms automatically:
1. Extract ground truth from `data.non_tensor_batch["reward_model"]["ground_truth"]`
2. Get rollout outputs from `batch.batch["responses"]`
3. Get entropies from `batch.batch["entropys"]` (or create zeros if not available)
4. Compute rewards using `compute_for_rollout(batch, return_dict=True)`, then let the configured `adv_estimator` build advantages

The trainer automatically detects `compute_for_rollout` and always runs `compute_advantage`; reward managers no longer compute advantages themselves.

## Multi-Model Output Parsing Support

All SRPO variants support multiple conversation formats:

- **Qwen format**: `<|im_start|>system\n...<|im_end|><|im_start|>user\n...<|im_end|>`
- **Llama3 format**: `<|start_header_id|>system<|end_header_id|>...<|eot_id|>`
- **DeepSeek format**: `system content` + `<｜tool▁calls▁begin｜>` + `user content` + `<｜Assistant｜>`

You can specify which parser to use via the `conversation_parser` parameter, or use `"auto"` to automatically detect the format.

## Notes

- All algorithms require `entropys` in the batch for segmentation. If not available, zeros will be used.
- The `repeat_times` parameter is automatically set from `config.actor_rollout_ref.rollout.n`
- All algorithms support both `__call__` (standard interface) and `compute_for_rollout` (direct interface) methods
- Multi-node multi-GPU inference is supported through Ray's distributed framework
- Segment reward computation requires an external API endpoint (for SRPO variants)
- All comments and docstrings are in English
