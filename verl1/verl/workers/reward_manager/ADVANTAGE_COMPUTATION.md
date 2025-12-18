# Advantage Computation in Reward Managers

## Current Behavior
- SRPO/RFLUX reward managers now **only compute rewards**. They return token-level rewards plus `sequence_rewards` via `compute_for_rollout(batch, return_dict=True)`.
- Advantages are computed by dedicated adv estimators registered in `core_algos.py` (`srpo`, `srpo2`, `srpo3`, `srpo4`, `rflux`).

## Trainer Flow
1. Reward stage collects RM scores if enabled.
2. After `entropys` are available, the trainer calls `reward_fn.compute_for_rollout(batch, return_dict=True)` to get rewards and `sequence_rewards`.
3. Trainer writes `token_level_scores`, `token_level_rewards`, and `sequence_rewards` into the batch.
4. `compute_advantage` always runs with the configured `adv_estimator` to build advantages/returns.

## Why Keep `compute_for_rollout`
- Some reward managers need `entropys`/segmentation to build rewards; `compute_for_rollout` lets them access the full `batch` without splitting arguments.
- Inputs/outputs now stay as a single `batch` object; no advantage is produced inside the reward manager.

## Configuration Tips
- Set `algorithm.adv_estimator` to the matching estimator for the reward manager (e.g., `srpo`, `srpo2`, `srpo3`, `srpo4`, `rflux`, or `grpo`).
- `compute_advantage` handles all advantage logic; no `_advantages_computed` marker is used anymore.
