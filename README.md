# gym_reacher Environment
This is an implementation of the reacher benchmark problem as an OpenAI Gym environment. 
This code is largely based on [pybullet-gym](https://github.com/benelot/pybullet-gym). It was simplified with the objective of understanding how to create custom Gym environments.

## Installing gym_reacher

```bash
pip install gym
pip install pybullet
pip install stable-baselines[mpi]
git clone https://github.com/PierreExeter/gym-reacher.git
cd reacher
pip install -e .
```

Important Note: *Do not* use `python setup.py install`.

## Test your installation
```bash
python scripts/test_gym_reacher.py
```

## Supported systems
Tested on Ubuntu 18.04 running Python 3.6.


