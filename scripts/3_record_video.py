import gym
import gym_reacher
import os
import time

from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy


env_id = 'Reacher3Dof-v0'
# env_id = 'CartPole-v1'

video_folder = '../results/tests/videos/'
os.makedirs(video_folder, exist_ok=True)


env = DummyVecEnv([lambda: gym.make(env_id)])
env.render(mode="human")  


def record_video(env_id, model, video_length=500, prefix='', video_folder=video_folder):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(
      env, 
      video_folder=video_folder,
      record_video_trigger=lambda step: step == 0, 
      video_length=video_length,
      name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)
    # time.sleep(1./30.) 

  # Close the video recorder
  eval_env.close()


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)


record_video(env_id, model, video_length=500, prefix='ppo2-reacher', video_folder=video_folder)
