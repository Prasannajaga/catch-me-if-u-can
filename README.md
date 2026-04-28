## CATCH ME IF YOU CAN

Well you heard the name this project started out of curiosity for how we handle the live frames in Reinforcement Learning


## CLI

```pythona
python -m train --timesteps 50000 --run-name first-experiment --check-env
python -m eval --episodes 5 --render
python -m play_live
```

which loss is important here

1. episode_length_curve
2. reward_curve
3. eval survival_rate
4. visual render behavior
5. PPO loss curves

## train.py Args Guide

1. `--run-name`
- Use to create a distinct run folder under `runs/`.
- Good for experiment tracking (`runs/<run-name>`).

2. `--timesteps`
- Total training steps for this invocation.
- Increase for longer training; use smaller values for incremental resume cycles.

3. `--n-envs`
- Number of parallel envs.
- Higher = faster data collection, but can change training dynamics.

4. `--seed`
- Random seed for reproducibility/comparability.

5. `--n-steps`
- Rollout horizon per env before PPO update.
- Larger = longer-horizon gradients, fewer updates.

6. `--batch-size`
- Minibatch size for PPO optimization.
- Larger can stabilize, smaller can add update noise.

7. `--learning-rate`
- PPO optimizer step size.
- Too high can destabilize; lower is safer for fine-tuning/resume.

8. `--gamma`
- Discount factor.
- Higher favors long-term survival behavior.

9. `--gae-lambda`
- GAE bias/variance trade-off.
- Usually near `0.95`; tuning affects advantage smoothness.

10. `--clip-range`
- PPO clip threshold.
- Lower = more conservative updates.

11. `--ent-coef`
- Entropy bonus weight.
- Higher encourages exploration; lower encourages policy commitment.

12. `--check-env`
- Runs Gym env sanity checks before training.
- Use when modifying env code.

13. `--render-test`
- Runs one random rendered episode before training.
- Use for quick visual env sanity check.

14. `--model-path`
- Primary final model output path.
- Default: `models/catchme_ppo.zip`.

15. `--log-dir`
- Explicit directory for logs/artifacts (`progress.csv`, monitor files, eval, checkpoints).

16. `--eval-freq`
- How often eval callback runs (in env timesteps).
- Lower = more frequent eval, more overhead.

17. `--eval-episodes`
- Episodes per eval callback run.
- Higher = more stable best-model selection, slower training.

18. `--checkpoint-freq`
- How often checkpoints are saved.
- Lower = more restore points, more disk I/O.

19. `--no-eval`
- Disable eval callback completely.
- Faster training, no `best_model.zip`.

20. `--no-checkpoint`
- Disable periodic checkpoint saving.

21. `--resume-from`
- Load existing PPO `.zip` and continue training from it.

22. `--reset-num-timesteps`
- On resume, reset timestep counter instead of continuing from loaded model's step count.

## Useful Patterns

1. Fresh training: set PPO hyperparams + run name.
2. Fast iterate: small `--timesteps`, `--no-eval`, `--no-checkpoint`.
3. Production run: eval/checkpoints enabled.
4. Fine-tune best model: `--resume-from ...` with conservative extra timesteps.
