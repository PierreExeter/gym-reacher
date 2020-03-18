import gym
import gym_reacher

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = make_vec_env('Reacher3Dof-v0', n_envs=4)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
# env.render(mode="human")

for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    print(i)
    print(action)
    print(obs)
    print(rewards)
    print(dones)
    print(info)

env.close()
    