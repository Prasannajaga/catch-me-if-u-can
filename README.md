## CATCH ME IF YOU CAN

Well you heard the name this project started out of curiosity for how we handle the live frames in Reinforcement Learning

## GOAL

you turn on the camera
one guy appear on the screen
you have to try to catch him if you can

this is simple idea behind this fun projects for this to be done
we have to do three things

these are the things i have done

1. create a custom environment where agent can play with
2. train the agent on the environment x, y coordinates
3. use openCV & media pipe to capture the hand signals

### constraints

well this is easier said than done
we can't train the model on live video frames
so we have to create a model that can play in the simulated environment
and then transfer that to the live environment like fake it till you make it

## CLI

```pythona
python -m train --timesteps 50000 --run-name first-experiment --check-env
python -m eval --episodes 5 --render
python -m play_live
```
