import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_reacher

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_vec_env


if __name__ == "__main__":


    def make_env(env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id)
            # Important: use a different seed for each environment
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init



    # env_id = 'CartPole-v1'
    env_id = 'Reacher3Dof-v0'
    # The different number of processes that will be used
    # PROCESSES_TO_TEST = [1, 2, 4, 8, 16] 
    PROCESSES_TO_TEST = [8, 16] 
    NUM_EXPERIMENTS = 3 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 5000
    # Number of episodes for evaluation
    EVAL_EPS = 20
    ALGO = PPO2

    # We will create one environment to evaluate the agent on
    eval_env = gym.make(env_id)

    # DummyVecEnv vs SubprocVecEnv
    reward_averages = []
    reward_std = []
    training_times = []
    total_procs = 0
    for n_procs in PROCESSES_TO_TEST:
        total_procs += n_procs
        print('Running for n_procs = {}'.format(n_procs))
        # Here we are using only one process even for n_env > 1
        # this is equivalent to DummyVecEnv([make_env(env_id, i + total_procs) for i in range(n_procs)])
        train_env = make_vec_env(env_id, n_envs=n_procs)

        rewards = []
        times = []

        for experiment in range(NUM_EXPERIMENTS):
            # it is recommended to run several experiments due to variability in results
            train_env.reset()
            model = ALGO('MlpPolicy', train_env, verbose=0)
            start = time.time()
            model.learn(total_timesteps=TRAIN_STEPS)
            times.append(time.time() - start)
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
            rewards.append(mean_reward)

        train_env.close()
        reward_averages.append(np.mean(rewards))
        reward_std.append(np.std(rewards))
        training_times.append(np.mean(times))

    # plot
    training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

    plt.figure()
    plt.subplot(1,2,1)
    plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2)
    plt.xlabel('Processes')
    plt.ylabel('Average return')
    plt.subplot(1,2,2)
    plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
    plt.xticks(range(len(PROCESSES_TO_TEST)),PROCESSES_TO_TEST)
    plt.xlabel('Processes')
    plt.ylabel('Training steps per second')
    plt.show()