## GOAL

you turn on the camera
one guy appear on the screen
you have to try to catch him if you can

this is simple idea behind this fun projects for this to be done
we have to do three things

these are the things i have done

1. create a custom environment where agent can play with
2. train the agent on the environment x, y coordinates
3. use openCV & media pipe to capture the hand signals''



## Constraints

well this is easier said than done
we can't train the model on live video frames
so we have to create a model that can play in the simulated environment
and then transfer that to the live environment like fake it till you make it


## how we Progressed so FAR

so far I check the value loss for picking the best model but later found out that
rewaards and episode steps is more important than the value (critic loss)

Learned a lessoon that given the constraint
the max_steps 600 we have to choose the bot whih doesn't caught <600 steps
so we compare survival and caught ratio to figure out which is the best model

heres couple of thing to understand

1. the more rewards the imrpoved survival rate
2. the more steps it take in single episode means the bot didn't caught so it lower the caught rate


## Eval & Metrics

utilmate-run-1M with episode 1000

survival : 70%
caught: 30%

Heres a config run:

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


ultimate-run-2M

survival: 20%
caught: 80%

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

so it appear it doens't matter the how long the run is
the better hyperparams and update with learning rate the good the model less likely getting caught