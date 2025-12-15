# Advantage Computation in Reward Managers

## Overview

The reward managers (GRPO, SRPO, SRPO2, SRPO3, SRPO4, RFLUX) compute advantages internally via the `compute_for_rollout` method. This document explains how the trainer ensures these advantages are not recomputed by the standard advantage estimator.

## How It Works

### 1. Detection Mechanism

The trainer automatically detects if a reward manager has the `compute_for_rollout` method:

```python
use_compute_for_rollout = hasattr(self.reward_fn, 'compute_for_rollout')
```

### 2. Computation Flow

#### Decoupled Mode (Default)

In decoupled mode, the flow is:

1. **Reward computation phase**: Standard reward computation (or skipped if using compute_for_rollout)
2. **Old log prob computation**: Compute `old_log_probs` and `entropys`
3. **Compute_for_rollout call**: After entropys are available, call `compute_for_rollout` which:
   - Computes segment-level rewards (for SRPO variants)
   - Computes sequence-level rewards
   - Computes advantages internally
   - Returns `(rewards_tensor, betaTensor, advTensor)`
4. **Advantage phase**: Check if advantages are already computed:
   - If `_advantages_computed` flag is set: Skip `compute_advantage` call
   - Otherwise: Call standard `compute_advantage` function

#### Bypass Mode

In bypass mode (when `rollout_correction.bypass_mode=True`):

1. **Rollout correction**: Apply rollout correction using `rollout_log_probs`
2. **Entropy check**: Check if entropys are available:
   - If available in batch: Use them directly
   - If not available: Compute them via `_compute_old_log_prob` (only for entropy)
3. **Compute_for_rollout call**: Same as decoupled mode
4. **Advantage phase**: Same check as decoupled mode

### 3. Protection Against Duplicate Computation

The trainer uses a flag `_advantages_computed` to mark when advantages have been computed by `compute_for_rollout`:

```python
# After compute_for_rollout
batch.batch["_advantages_computed"] = True

# In advantage phase
advantages_already_computed = batch.batch.get("_advantages_computed", False)
if use_compute_for_rollout and advantages_already_computed:
    # Skip compute_advantage call
    # Just set the values
else:
    # Call standard compute_advantage
    batch = compute_advantage(...)
```

## Code Flow

```
┌─────────────────────────────────────┐
│  Reward Computation Phase          │
│  (Standard reward or skipped)      │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Old Log Prob Computation            │
│  - Computes entropys                 │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Compute_for_rollout (if available) │
│  - Uses entropys from old_log_prob  │
│  - Computes rewards + advantages   │
│  - Sets _advantages_computed = True │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Advantage Phase                    │
│  - Checks _advantages_computed flag │
│  - If True: Skip compute_advantage  │
│  - If False: Call compute_advantage │
└─────────────────────────────────────┘
```

## Key Points

1. **No Duplicate Computation**: The `_advantages_computed` flag ensures `compute_advantage` is never called when advantages are already computed by `compute_for_rollout`.

2. **Entropy Availability**: 
   - In decoupled mode: Entropys are computed in `old_log_prob` phase
   - In bypass mode: Entropys are checked/computed before `compute_for_rollout` call

3. **Backward Compatibility**: If a reward manager doesn't have `compute_for_rollout`, the standard flow continues with `compute_advantage`.

4. **Multi-Node Support**: All computations work correctly in distributed Ray environments.

## Verification

To verify that advantages are not being recomputed:

1. Check logs: When using `compute_for_rollout`, you should see:
   - "Start computing rewards"
   - "Start computing advantages" (from reward manager)
   - But NOT see the standard advantage computation logs

2. Check batch contents: After reward computation, `batch.batch["_advantages_computed"]` should be `True`.

3. Monitor performance: Using `compute_for_rollout` should be faster than computing rewards and advantages separately.

## Configuration

No special configuration is needed. The trainer automatically detects and uses `compute_for_rollout` if available. Simply specify the reward manager name:

```yaml
reward_manager:
  source: "register"
  name: "grpo"  # or "srpo", "srpo2", etc.
```

The trainer will automatically:
- Detect `compute_for_rollout` method
- Call it at the right time (after entropys are available)
- Skip standard advantage computation
- Use the computed advantages directly
