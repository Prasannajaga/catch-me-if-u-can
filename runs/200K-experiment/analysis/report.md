# Training Analysis Report

**Run Directory**: `runs/200K-experiment`

## Summary
- Episodes: 6682
- Total Timesteps: 200668
- Best Episode Reward: 22.05
- Best Rolling Reward: 5.48
- Final Rolling Reward: 2.95
- Best Rolling Length: 39.86
- Final Rolling Length: 36.32
- Reward Improvement: 213.2%
- Length Improvement: 48.6%

## Diagnosis
- **GOOD_LEARNING**: No
- **PLATEAU**: No
- **COLLAPSE**: Yes
- **NO_LEARNING**: No
- **POSSIBLE_EDGE_EXPLOIT**: No
- **UNSTABLE_PPO**: No
- **VALUE_FUNCTION_BAD**: No
- **EXPLORATION_COLLAPSE**: No

## Hyperparameter Recommendations
- use best checkpoint, not final model
- lower learning_rate
- lower clip_range
- add EvalCallback and save best model
- stop training when eval reward stops improving

## Next Experiment Suggestion
Agent suffered a performance collapse. Use earlier checkpoints or reduce update step sizes.
