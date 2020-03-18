import time
import gym
import numpy as np
import gym_reacher

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy

# compare performance between single and multi process

if __name__ == '__main__':

    # note: the evaluate_policy utility from Stable baselines doesn work here
    # so I need to use the evaluate function described here instead

    def make_env(env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init

    env_id = "Reacher3Dof-v0"
    num_cpu = 4  # Number of processes to use

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = make_vec_env(env_id, n_envs=4, vec_env_cls=SubprocVecEnv)

    model = ACKTR(MlpPolicy, env, verbose=0)

    def evaluate(model, num_steps=1000):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward
        """
        episode_rewards = [[0.0] for _ in range(env.num_envs)]
        obs = env.reset()
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            actions, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, rewards, dones, info = env.step(actions)
            
            # Stats
            for i in range(env.num_envs):
                episode_rewards[i][-1] += rewards[i]
                if dones[i]:
                    episode_rewards[i].append(0.0)

        mean_rewards =  [0.0 for _ in range(env.num_envs)]
        n_episodes = 0
        for i in range(env.num_envs):
            mean_rewards[i] = np.mean(episode_rewards[i])     
            n_episodes += len(episode_rewards[i])   

        # Compute mean reward
        mean_reward = round(np.mean(mean_rewards), 1)
        print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

        return mean_reward


    # Random Agent, before training
    mean_reward_before_train = evaluate(model, num_steps=1000)

    n_timesteps = 25000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))
    
    # Evaluate the trained agent
    mean_reward = evaluate(model, num_steps=10000)


    # Single Process RL Training
    single_process_model = ACKTR(MlpPolicy, DummyVecEnv([lambda: gym.make(env_id)]), verbose=0)

    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time_single, n_timesteps / total_time_single))

    print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))

    # Evaluate the trained agent
    mean_reward = evaluate(model, num_steps=10000)
