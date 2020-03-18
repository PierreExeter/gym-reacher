import gym
import gym_reacher

from stable_baselines.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines.common.vec_env import DummyVecEnv


# env = gym.make("Pendulum-v0")
env = gym.make("Reacher3Dof-v0")

print("standard Gym environment:")

print("action space: ")
print(env.action_space)
print(env.action_space.low)
print(env.action_space.high)

print("observation space: ")
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)


obs = env.reset()
for _ in range(10):
  action = env.action_space.sample()
  obs, reward, _, _ = env.step(action)

  print(obs, reward)

############

env = DummyVecEnv([lambda: env])

print("DummyVec environment:")


obs = env.reset()
for _ in range(10):
  action = [env.action_space.sample()]
  obs, reward, _, _ = env.step(action)

  print(obs, reward)

##########

# normalise observations and reward
env = VecNormalize(env)


print("VecNormalize environment:")

obs = env.reset()
for _ in range(10):
  action = [env.action_space.sample()]
  obs, reward, _, _ = env.step(action)

  print(obs, reward)

