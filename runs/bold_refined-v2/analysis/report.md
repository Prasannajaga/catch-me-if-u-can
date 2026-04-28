# Training Analysis Report

**Run Directory**: `runs/bold_refined-v2`

## Summary
- Episodes: 10128
- Total Timesteps: 303005
- Best Episode Reward: 21.59
- Best Rolling Reward: 4.78
- Final Rolling Reward: 2.78
- Best Rolling Length: 39.18
- Final Rolling Length: 35.30
- Reward Improvement: 210.0%
- Length Improvement: 52.0%

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
