# gym_reacher Environment
This is an implementation of the reacher benchmark problem as an OpenAI Gym environment. 
This code is largely based on [pybullet-gym](https://github.com/benelot/pybullet-gym). It was simplified with the objective of understanding how to create custom Gym environments.

## Installing gym_reacher

Install [Gym](https://github.com/openai/gym).

```bash
pip install gym
```

Install [Pybullet](https://pypi.org/project/pybullet/).
 
```bash
pip install pybullet
```

Install [Stable-baselines](https://github.com/pirobot/stable-baselines).

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines[mpi]
```

Install [Tensorflow 1.14](https://www.tensorflow.org/). Stable-baselines does not yet support Tensorflow 2.

```bash
pip install tensorflow-gpu==1.14
```

Install gym-reacher.

```bash
git clone https://github.com/PierreExeter/gym-reacher.git
cd gym-reacher
pip install -e .
```

Important Note: *Do not* use `python setup.py install`.

## Test your installation
```bash
python scripts/test_gym_reacher.py
```

## Train models
```bash
python scripts/train_all.py
```

## Test models
```bash
python scripts/enjoy_all.py
```

## Plot results
```bash
cd results
tensorboard --logdir=A2C:tensorboard_logs/A2C/, ACKTR:tensorboard_logs/ACKTR/, DDPG:tensorboard_logs/DDPG/, PPO1:tensorboard_logs/PPO1/, PPO2:tensorboard_logs/PPO2/, SAC:tensorboard_logs/SAC/, TRPO:tensorboard_logs/TRPO/, TD3:tensorboard_logs/TD3/
```

## Supported systems
Tested on:
- Ubuntu 18.04 
- Python 3.6
- Tensorflow 1.14


