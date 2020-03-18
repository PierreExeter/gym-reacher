import gym
import numpy as np
import pandas as pd
from pathlib import Path
import gym_reacher
import time

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



env_list = [
    'Reacher1Dof-v0',
    'Reacher2Dof-v0',
    'Reacher3Dof-v0',
    'Reacher4Dof-v0',
    'Reacher5Dof-v0',
    'Reacher6Dof-v0',
]

# for env_name in env_list:

# create environment
# env_name = 'Reacher1Dof-v0'
# env_name = 'Reacher2Dof-v0'
# env_name = 'Reacher3Dof-v0'
# env_name = 'Reacher4Dof-v0'
# env_name = 'Reacher5Dof-v0'
env_name = 'Reacher6Dof-v0'

env = gym.make(env_name)
# The algorithms require a vectorized environment to run
# env = DummyVecEnv([lambda: env])

res_dir = "../results/"+env_name
tensorboard_dir = res_dir+"/tensorboard_logs/"
reward_dir = res_dir+"/reward_vs_steps/"
trained_dir = res_dir+"/trained_models/"
time_dir = res_dir+"/time/"
acc_dir = res_dir+"/accuracy/"

Path(acc_dir).mkdir(parents=True, exist_ok=True)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(
    n_actions), sigma=0.1 * np.ones(n_actions))

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


# model_list = [model_list[5], model_list[5]]
# algo_list = [algo_list[5], algo_list[5]]

# env.render(mode="human")  # this needs to be placed BEFORE env.reset()

acc_list = []
av_reach_time_list = []

for model, algo in zip(model_list, algo_list):
    print(algo)
    model = model.load(trained_dir+algo)

    episode_nb = 0
    successful_episodes = 0
    timestep_successful_ep = []

    while episode_nb <= 100:
    # for episode_nb in range(20):

        obs = env.reset()           
        
        # only run an episode if the target is reachable
        robot_reach, dist_target_origin, dist_ft_t = env.extra_info()
        # print("distance origin to target:", dist_target_origin)
        # print("robot reach:", robot_reach)
        # print("distance fingertip - target:", dist_ft_t)
        
        if dist_target_origin <= robot_reach:
            print("episode: ", episode_nb)
            episode_nb += 1

            # for t in range(100):
            t = 0
            while t <= 100:

                # print(t)

                action, _states = model.predict(obs) 
                obs, reward, done, info = env.step(action)

                # env.render()
                # time.sleep(1./30.) 

                if done:
                    successful_episodes += 1
                    timestep_successful_ep.append(t)

                    # robot_reach, dist_target_origin, dist_ft_t = env.extra_info()
                    # print("distance origin to target:", dist_target_origin)
                    # print("robot reach:", robot_reach)
                    # print("distance fingertip - target:", dist_ft_t)

                    break
                
                t += 1
            
        
        else:
            print("Target not reachable, a new target will be generated.")
  
    success_ratio = successful_episodes/episode_nb
    acc_list.append(success_ratio)
    print("nb episodes: ", episode_nb)   
    print("nb successful episode: ", successful_episodes)   
    print("sucess ratio:", success_ratio)

    if successful_episodes != 0:
        average_reach_time = sum(timestep_successful_ep)/successful_episodes
        av_reach_time_list.append(average_reach_time)
        print("average_reach_time:", average_reach_time)
        print("timestep_successful_ep: ", timestep_successful_ep)

    df = pd.DataFrame(list(zip(algo_list, acc_list, av_reach_time_list)), columns=['algo', 'success ratio', 'average reach time'])
    df.to_csv(acc_dir+'accuracy_reach.csv', index=False)

env.close()


