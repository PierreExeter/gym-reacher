import gym
import gym_reacher
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('Reacher3Dof-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)

# train
model.learn(total_timesteps=10000)

# or in one line
# model = PPO2('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

# test
# I need to instanciate a new environment to be able to render (vectorized environment by stable baselines)

env.close()
env = gym.make('Reacher3Dof-v0')
env.render(mode="human")   


for episode in range(20):
    obs = env.reset()             
    rewards = []
    
    for t in range(100):
        # action = env.action_space.sample()  
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action) 
        print ("Episode: {0}\n Time step: {1}\n Action: {2}\n Observation: {3}\n Reward: {4}\n done: {5} \n info: {6}".format(episode, t, action, obs, reward, done, info))
        rewards.append(reward)
        time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  
    
env.close()