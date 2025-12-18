# Reward Manager Algorithms

This directory contains reward manager implementations that compute rule-based rewards.
Advantage computation is now handled by dedicated adv estimators (e.g., `grpo`,
`srpo`, `srpo2`, `srpo3`, `srpo4`, `rflux`) registered in `core_algos.py`.

## Available Algorithms

### 1. GRPO (Group Relative Policy Optimization)
- **Registration name**: `grpo`
- **Description**: Computes sequence-level rewards and performs within-group normalization to use relative advantages instead of absolute rewards.
- **Key features**:
  - Only uses sequence-level rewards
  - Normalizes rewards within groups (repeat_times samples)
  - Broadcasts normalized advantages to all timesteps

### 2. SRPO (Segment-based Reward Policy Optimization)
- **Registration name**: `srpo`
- **Description**: Computes both segment-level and sequence-level rewards, then combines them with normalization.
- **Key features**:
  - Uses entropy-based segmentation
  - Computes segment-level rewards
  - Combines segment and sequence rewards

### 3. SRPO2
- **Registration name**: `srpo2`
- **Description**: Similar to SRPO but uses DeepSeek conversation parsing and subtracts mean from beta.
- **Key differences from SRPO**:
  - Uses `parse_deepseek_conversation` instead of `parse_qwen_conversation`
  - Beta calculation: `tz_scores - mean` (centered)

### 4. SRPO3
- **Registration name**: `srpo3`
- **Description**: Similar to SRPO2 but adds mean segment reward to sequence reward.
- **Key differences from SRPO2**:
  - `rollout_raw_reward = seq_reward + mean_seqReward`

### 5. SRPO4
- **Registration name**: `srpo4`
- **Description**: Similar to SRPO but uses within-group std for segment normalization.
- **Key differences from SRPO**:
  - Uses within-group std for segment normalization instead of fixed scv

### 6. RFLUX
- **Registration name**: `rflux`
- **Description**: Combines sequence reward with mean segment rewards, uses only beta as advantage.
- **Key features**:
  - `rollout_raw_reward = seq_reward + mean(segment_rewards)`
  - Only uses beta (sequence-level) as advantage, no segment-level component
  - Uses DeepSeek conversation parsing

## Configuration Parameters

When running `main_ppo.py`, you need to specify the following parameters:

### Basic Configuration

```yaml
reward_manager:
  source: "register"  # Use registered reward managers
  name: "grpo"  # or "srpo", "srpo2", "srpo3", "srpo4", "rflux"
  
reward_model:
  reward_kwargs:
    alpha: 0.5  # Weight parameter (for compatibility, not used in all algorithms)
    max_branches: 5  # Maximum branches for segmentation
    min_gap: 10  # Minimum gap for segmentation
    endpoint: "http://your-endpoint:port/v1/step_rewards"  # Optional: for segment rewards
```

### Example Configuration

```yaml
# For GRPO
reward_manager:
  source: "register"
  name: "grpo"

reward_model:
  reward_kwargs: {}

# For SRPO with custom parameters
reward_manager:
  source: "register"
  name: "srpo"

reward_model:
  reward_kwargs:
    max_branches: 5
    min_gap: 10
    endpoint: "http://localhost:4997/v1/step_rewards"
```

### Multi-Node Multi-GPU Support

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
4. Compute rewards and advantages using `compute_for_rollout` method

The trainer will automatically detect if `compute_for_rollout` method exists and use it, otherwise falls back to the standard `__call__` method.

## Notes

- All algorithms require `entropys` in the batch for segmentation. If not available, zeros will be used.
- The `repeat_times` parameter is automatically set from `config.actor_rollout_ref.rollout.n`
- All algorithms support both `__call__` (standard interface) and `compute_for_rollout` (direct interface) methods
- Multi-node multi-GPU inference is supported through Ray's distributed framework
