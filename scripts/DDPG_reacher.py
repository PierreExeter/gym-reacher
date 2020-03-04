import gym
import time
import numpy as np
import random
import time
import csv
import gym_reacher
import pandas as pd
from pathlib import Path

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, ACER, DQN, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

# create environment
env_name = 'Reacher1Dof-v0'
# env_name = 'Reacher2Dof-v0'
# env_name = 'Reacher3Dof-v0'
# env_name = 'Reacher4Dof-v0'
# env_name = 'Reacher5Dof-v0'
# env_name = 'Reacher6Dof-v0'

# env = make_vec_env(env_name, n_envs=4)
env = gym.make(env_name)  # DDPG requires a non-vectorized environment

res_dir = "../results_"+env_name
tensorboard_dir = res_dir+"/tensorboard_logs"
reward_dir = res_dir+"/reward_vs_steps"

Path(res_dir).mkdir(parents=True, exist_ok=True)
Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
Path(reward_dir).mkdir(parents=True, exist_ok=True)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise_td3 = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# the noise objects for DDPG
param_noise = None
action_noise_ddpg = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


model_list = [
        # A2C(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/A2C/"), 
        # ACER(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/ACER/"), 
        # ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/ACKTR/"), 
        # DDPG(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/DDPG/", param_noise=param_noise, action_noise=action_noise_ddpg),
        # DQN(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/DDQN/"), 
        # DQN(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/DQN/", policy_kwargs=dict(dueling=False)), 
        # DQN(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/DDQN_PER/", prioritized_replay=True), 
        PPO1(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/PPO1/"), 
        # PPO2(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/PPO2/"), 
        SAC(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/SAC/"), 
        TD3(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/TD3/", action_noise=action_noise_td3), 
        TRPO(MlpPolicy, env, verbose=1, tensorboard_log="../results/tensorboard_logs/TRPO/"), 
        ]

algo_list = [
        # "A2C",
        # "ACER",
        # "ACKTR",
        # "DDPG",
        # "DDQN",
        # "DQN",
        # "DDQN_PER",
        "PPO1",
        # "PPO2",
        "SAC",
        "TD3",
        "TRPO",
        ]

training_time_list = []

for model, algo in zip(model_list, algo_list):
    print(model)

    start = time.time()
    model.learn(total_timesteps=10000)
    end = time.time()
    training_time_list.append((end-start)*1000)
    model.save("../results/trained_models/"+algo)


df = pd.DataFrame(list(zip(algo_list, training_time_list)), columns=['algo', 'train_time (ms)'])
df.to_csv('../results/train_time_'+algo+'.csv', index=False)

env.close()
