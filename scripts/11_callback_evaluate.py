import gym
import gym_reacher

from stable_baselines import SAC
from stable_baselines.common.callbacks import EvalCallback

# Evaluate periodically the performance of an agent, using a separate test environment and save the best one


# Separate evaluation env
eval_env = gym.make('Reacher3Dof-v0')

# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='../results/tests/logs/',
    log_path='../results/tests/logs/', 
    eval_freq=500,
    deterministic=True, 
    render=False)

model = SAC('MlpPolicy', 'Reacher3Dof-v0')
model.learn(5000, callback=eval_callback)