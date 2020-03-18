import numpy as np
import gym
import gym_reacher

from stable_baselines import SAC
from stable_baselines.common.evaluation import evaluate_policy


eval_env = gym.make('Reacher3Dof-v0')
default_model = SAC('MlpPolicy', 'Reacher3Dof-v0', verbose=0, seed=0).learn(8000)

mean_reward, std_reward = evaluate_policy(default_model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

tuned_model = SAC('MlpPolicy', 'Reacher3Dof-v0', batch_size=256, verbose=0, policy_kwargs=dict(layers=[256, 256]), seed=0).learn(8000)

mean_reward, std_reward = evaluate_policy(tuned_model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")