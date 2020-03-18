import imageio
import numpy as np
import gym
import gym_reacher

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('Reacher3Dof-v0')
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

# model = A2C("MlpPolicy", "LunarLander-v2").learn(100000)

images = []
obs = model.env.reset()
# img = model.env.render(mode='human')
img = model.env.render(mode='rgb_array')
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    # img = model.env.render(mode='human')
    img = model.env.render(mode='rgb_array')

imageio.mimsave('../results/tests/reacher3dof.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)