import numpy as np
import gym
import gym_reacher

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy


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


# original_env = gym.make("Pendulum-v0")
original_env = gym.make("Reacher6Dof-v0")


print("action space boundaries: ")
print(original_env.action_space.low)
print(original_env.action_space.high)

print("example of random actions:")
for _ in range(10):
  print(original_env.action_space.sample())


# env = NormalizeActionWrapper(gym.make("Pendulum-v0"))
env = NormalizeActionWrapper(gym.make("Reacher6Dof-v0"))

print("action space boundaries: ")
print(env.action_space.low)
print(env.action_space.high)

print("example of random actions:")
for _ in range(10):
  print(env.action_space.sample())
