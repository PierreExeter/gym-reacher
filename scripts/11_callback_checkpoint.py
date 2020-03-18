import gym_reacher

from stable_baselines import SAC
from stable_baselines.common.callbacks import CheckpointCallback


# Save a checkpoint every 1000 steps

checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path='../results/tests/logs/',
    name_prefix='rl_model')

model = SAC('MlpPolicy', 'Reacher3Dof-v0')
model.learn(2000, callback=checkpoint_callback)