# Catch Me If You Can

A small reinforcement learning project for training and evaluating an agent in a live environment.

## Quick Start

```bash
python -m train --timesteps 50000 --run-name first-experiment --check-**env**
python -m eval --episodes 5 --render
python -m play_live
```



## How We Did It

We built a custom environment, trained a PPO agent, and iterated through short training/evaluation cycles while checking live behavior.

For detailed notes and progress, see [PROGRESS.md](PROGRESS.md).
