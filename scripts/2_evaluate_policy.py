import gym
import gym_reacher

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy

eval_env = gym.make('Reacher3Dof-v0')
env = gym.make('Reacher3Dof-v0')

# env = gym.make('CartPole-v1')
model = PPO2(MlpPolicy, env, verbose=0)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=300)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=10000)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=300)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
