## Goal

You turn on the camera, one person appears on screen, and the agent tries to catch them.

That is the core idea of this project. To make it work, we focused on three things:

1. Create a custom environment where the agent can learn.
2. Train the agent on environment `x, y` movement patterns.
3. Use OpenCV + MediaPipe to capture live hand signals.

## Constraints

This is harder than it sounds. We cannot directly train on live video frames.

So the approach is:

- Train in a simulated environment first.
- Transfer that behavior into the live setup.

## How We Progressed So Far

At first, I focused on value loss to choose the best model. Later, I found rewards and episode steps were more useful than critic loss alone.

Given the `max_steps = 600` constraint, the better bot is the one that avoids getting caught before 600 steps. So we compare survival rate and catch rate to select the best model.

Key takeaways:

1. Higher rewards usually mean improved survival rate.
2. More steps per episode usually mean lower catch rate (the bot survives longer).

## Eval and Metrics

### `ultimate-run-1M` (1000 eval episodes)

- Survival: 70%
- Caught: 30%

Config:

```json
{
  "run_name": "ultimate_run-1M",
  "timesteps": 1000000,
  "n_envs": 4,
  "seed": 7,
  "n_steps": 4096,
  "batch_size": 512,
  "learning_rate": 0.0005,
  "gamma": 0.995,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "ent_coef": 0.001,
  "check_env": false,
  "render_test": false
}
```

### `ultimate-run-2M`

- Survival: 20%
- Caught: 80%

Config:

```json
{
  "run_name": null,
  "timesteps": 2000000,
  "n_envs": 4,
  "seed": 142,
  "n_steps": 2048,
  "batch_size": 512,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.15,
  "ent_coef": 0.005,
  "check_env": false,
  "render_test": false,
  "model_path": "/home/prasanna/coding/catch-me-if-you-can/models/catchme_ppo.zip",
  "log_dir": "runs/ultimate-run-2M-seed142",
  "eval_freq": 10000,
  "eval_episodes": 10,
  "checkpoint_freq": 100000,
  "no_eval": false,
  "no_checkpoint": false
}
```

This showed that longer training alone does not guarantee better results. Better hyperparameters and learning-rate updates matter more for reducing catch rate.

## Resume Improvements

## First Resume

### `ultimate-run-1M`

- Episodes: 100
- Mean reward: 139.23
- Mean steps: 399.08
- Catch rate: 36.00%
- Survival rate: 64.00%

### `ultimate_run_1m_resume_v1`

- Episodes: 100
- Mean reward: 184.12
- Mean steps: 481.97
- Catch rate: 21.00%
- Survival rate: 79.00%

We resumed training from `runs/ultimate_run-1M/catchme_ppo.zip` (our previous best checkpoint). It improved by about 15%, which was a strong jump.

## Second Resume

### `ultimate_run_1m_resume_v2`

- Episodes: 300
- Mean reward: 186.05
- Mean steps: 477.54
- Catch rate: 21.67%
- Survival rate: 78.33%

### `ultimate_run_1m_resume_v1`

- Episodes: 300
- Mean reward: 173.96
- Mean steps: 456.77
- Catch rate: 25.33%
- Survival rate: 74.67%

Another solid improvement.

### Final improvement 

### ultimate_run_1m_resume_v3_95push 
episodes:       300
mean_reward:    189.08
mean_steps:     550.92
catch_rate:     8.67%
survival_rate:  91.33%



## Config Comparison (`200K-experiment` → `ultimate_run-1M`)

### 1) Outcome Metrics

| Metric | 200K-experiment | ultimate_run-1M | Change |
|---|---:|---:|---:|
| Eval Mean Reward | -114.66 | 91.02 | **+205.68** |
| Survival Rate | 0.0% | 68.6% | **+68.6 pp** |
| Catch Rate | 100.0% | 31.4% | **-68.6 pp** |
| Eval Mean Steps | 40.24 | 423.19 | **+382.95** |
| Final Train Mean Reward | 3.69 | 121.71 | **+118.03** |
| Final Value Loss | 2.84 | 20.87 | +18.03 (higher, but performance still much better) |
| Total Timesteps (actual) | 200,704 | 1,015,808 | **+815,104 (~5.06x)** |

---

### 2) Arg Changes and Observed Impact

| Arg | 200K-experiment | ultimate_run-1M | Observed impact in this comparison |
|---|---:|---:|---|
| `timesteps` | 200000 | 1000000 | Biggest likely driver. Much longer training strongly aligns with major gains in reward/survival. |
| `n_steps` | 1024 | 4096 | Larger rollout horizon may help longer-term credit assignment, but can reduce update frequency. |
| `batch_size` | 256 | 512 | More stable gradient estimates; typically smoother updates. |
| `learning_rate` | 3e-4 | 5e-4 | More aggressive updates; can speed learning or destabilize depending on setup. |
| `gamma` | 0.99 | 0.995 | More future-focused objective; can help long survival behavior. |
| `ent_coef` | 0.01 | 0.001 | Much less exploration noise late in training; helps policy exploit learned survival behavior. |
| `check_env` | true | false | No direct performance effect (debug/check only). |
| `render_test` | false | false | No change. |

---

### 3) Unchanged Args

| Arg | Value |
|---|---:|
| `n_envs` | 4 |
| `seed` | 7 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |

---

### Important Note

This is a **multi-parameter change**, so exact per-argument causality is not isolated.  
The table above shows the most likely contribution based on RL behavior + observed outcomes.