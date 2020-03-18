import gym
import numpy as np
import pandas as pd
from pathlib import Path
import gym_reacher

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import time



env_list = [
    'Reacher1Dof-v0',
    'Reacher2Dof-v0',
    'Reacher3Dof-v0',
    'Reacher4Dof-v0',
    'Reacher5Dof-v0',
    'Reacher6Dof-v0',
]

for env_name in env_list:

    # create environment
    # env_name = 'Reacher1Dof-v0'
    # env_name = 'Reacher2Dof-v0'
    # env_name = 'Reacher3Dof-v0'
    # env_name = 'Reacher4Dof-v0'
    # env_name = 'Reacher5Dof-v0'
    # env_name = 'Reacher6Dof-v0'

    env = gym.make(env_name)
    # env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    res_dir = "../results/"+env_name
    tensorboard_dir = res_dir+"/tensorboard_logs/"
    reward_dir = res_dir+"/reward_vs_steps/"
    trained_dir = res_dir+"/trained_models/"
    time_dir = res_dir+"/time/"
    acc_dir = res_dir+"/accuracy/"

    Path(res_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(reward_dir).mkdir(parents=True, exist_ok=True)
    Path(trained_dir).mkdir(parents=True, exist_ok=True)
    Path(time_dir).mkdir(parents=True, exist_ok=True)
    Path(acc_dir).mkdir(parents=True, exist_ok=True)

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model_list = [
            A2C(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir+"A2C/"), 
            ACKTR(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir+"ACKTR/"), 
            DDPG(mlp_ddpg, env, verbose=1, tensorboard_log=tensorboard_dir+"DDPG/"),
            PPO1(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir+"PPO1/"),
            PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir+"PPO2/"), 
            SAC(mlp_sac, env, verbose=1, tensorboard_log=tensorboard_dir+"SAC/"), 
            TRPO(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir+"TRPO/"),
            TD3(mlp_td3, env, action_noise=action_noise, verbose=1, tensorboard_log=tensorboard_dir+"TD3/"),
    ]

    algo_list = ['A2C', 'ACKTR', 'DDPG', 'PPO1', 'PPO2', 'SAC', 'TRPO', 'TD3']

    training_time_list = []
    for model, algo in zip(model_list, algo_list):
        print(model)

        start = time.time()
        model.learn(total_timesteps=1000000)
        end = time.time()
        training_time_list.append((end-start)*1000)
        model.save(trained_dir+algo)


    df = pd.DataFrame(list(zip(algo_list, training_time_list)), columns=['algo', 'train_time (ms)'])
    df.to_csv(time_dir+'train_time.csv', index=False)


env.close()
