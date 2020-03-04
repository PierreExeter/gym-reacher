#!/usr/bin/env python3

import gym
import gym_gazebo
import jaco_reach
import time
import numpy
import random
import time
import csv

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


env = gym.make('JacoReach-v0')
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


print('********************** after creating env ********************')

print("launching the world...")
#gz loaing issues, let user start the learning
input("hit enter when gazebo is loaded...")
env.set_physics_update(0.0001, 10000)
input("hit enter when gazebo is loaded...")
env.set_goal([0.167840578046, 0.297489331432, 0.857454500127])

print('********************** after set goal ********************')


# The noise objects for TD3
# n_actions = env.action_space.n
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model_list = [
        A2C(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/A2C/"), 
        ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/ACKTR/"), 
        PPO2(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/PPO2/"), 
        # TRPO(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/TRPO/"),
]

algo_list = [
        'A2C', 
        'ACKTR', 
        'PPO2', 
        # 'TRPO'
        ]
# TEST
# model_list = [model_list[5], model_list[5]]
# algo_list = [algo_list[5], algo_list[5]]


print('********************** before test ********************')

for model, algo in zip(model_list, algo_list):
    print(algo)
    model = model.load("../results/trained_models/"+algo)
    obs = env.reset()

    for i in range(500):
        if i % 100 == 0:
            print(i)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.2)
         

    print('********************** after test ********************')

env.close()


