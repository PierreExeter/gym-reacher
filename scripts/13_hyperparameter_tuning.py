import gym
import gym_reacher
import time
import os
import numpy as np
import pandas as pd
import optuna
import yaml

from pathlib import Path


from stable_baselines import A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.ddpg.policies import MlpPolicy as mlp_ddpg
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize


class NormalizeActionWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    # Retrieve the action space
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
    # Retrieve the max/min values
    self.low, self.high = action_space.low, action_space.high

    # We modify the action space, so all actions will lie in [-1, 1]
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

    # Call the parent constructor, so we can access self.env later
    super(NormalizeActionWrapper, self).__init__(env)
  
  def rescale_action(self, scaled_action):
      """
      Rescale the action from [-1, 1] to [low, high]
      (no need for symmetric action space)
      :param scaled_action: (np.ndarray)
      :return: (np.ndarray)
      """
      return self.low + (0.5 * (scaled_action + 1.0) * (self.high -  self.low))

  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    # Rescale action from [-1, 1] to original [low, high] interval
    rescaled_action = self.rescale_action(action)
    obs, reward, done, info = self.env.step(rescaled_action)
    return obs, reward, done, info


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env = NormalizeActionWrapper(env_id)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def evaluate_multi(model, env, num_steps=1000):
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
        actions, _states = model.predict(obs, deterministic=True)
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


def sample_ppo2_params(trial):
    """
    Sampler for PPO2 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    lam = trial.suggest_categorical('lam', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
    cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        'gamma': gamma,
        'n_steps': n_steps,
        'ent_coef': ent_coef,
        'learning_rate': learning_rate,
        'lam': lam,
        'nminibatches': nminibatches,
        'noptepochs': noptepochs,
        'cliprange': cliprange
    }


n_env = 4
env_id = 'Reacher3Dof-v0'

def create_env(env_id):

    # env = gym.make(env_id)
    # env = DummyVecEnv([lambda: env])
    env = DummyVecEnv([make_env(env_id, i, seed=0) for i in range(n_env)])
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(n_env)])
    # env = VecNormalize(env)  
    

    # eval_env = gym.make(env_id)
    # eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = DummyVecEnv([make_env(env_id, i, seed=0) for i in range(n_env)])
    # eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(n_env)])
    # eval_env = VecNormalize(eval_env)  
    

    return env, eval_env


def optimize_agent(trial):
    """ Train the model and optimise
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """

    model_params = sample_ppo2_params(trial)
    print(model_params)

    env, eval_env = create_env(env_id)
    model = PPO2(MlpPolicy, env, verbose=0, **model_params)
    start_train = time.time()
    model.learn(total_timesteps=10000)
    end_train = time.time()
    print("opti train time (s): ", end_train-start_train)

    # mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)  # cannot use with multiprocess
    # mean_reward = evaluate_single(model, eval_env, num_steps=1000)
    start_eval = time.time()
    mean_reward = evaluate_multi(model, eval_env, num_steps=2000)
    end_eval = time.time()
    print("opti eval time (s): ", end_eval-start_eval)

    return -1 * mean_reward



if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(optimize_agent, n_trials=100, n_jobs=-1)

    best_params = study.best_params
    del best_params['batch_size']    # batch_size is not a PPO2 parameter
    print("best params: ", best_params)
    # print("best value: ", study.best_value)
    # print("best best trial: ", study.best_trial)

    with open('../results/hyperparameter.yml', 'w') as outfile:
        yaml.dump(best_params, outfile)

   
    env, eval_env = create_env(env_id)
    model = PPO2(MlpPolicy, env, verbose=1, **best_params)
    start = time.time()
    model.learn(total_timesteps=100000)
    end = time.time()

    print("training time (s): ", end-start)

    # mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)  # cannot use with multiprocess
    # mean_reward = evaluate_single(model, eval_env, num_steps=1000)
    mean_reward = evaluate_multi(model, eval_env, num_steps=10000)
   