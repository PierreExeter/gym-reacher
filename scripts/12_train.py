import gym
import gym_reacher
import time
import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy


# Create log dir
log_dir = "../results/tests/"
os.makedirs(log_dir, exist_ok=True)


env = gym.make("Reacher3Dof-v0")
env = Monitor(env, filename=log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)             # normalise observations and reward (action space already normalised)

print("action space: ")
print(env.action_space)
print(env.action_space.low)
print(env.action_space.high)

print("observation space: ")
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("default_params")


with open('../results/hyperparameter.yml') as file:
    hyper = yaml.load(file, Loader=yaml.FullLoader)

print(hyper)

model = PPO2(MlpPolicy, env, verbose=1, **hyper)
model.learn(total_timesteps=10000)
model.save("tuned_params")