import gym
import gym_reacher
import time
from stable_baselines.common.env_checker import check_env


# env = gym.make('Reacher1Dof-v0')
# env = gym.make('Reacher2Dof-v0')
# env = gym.make('Reacher3Dof-v0')
# env = gym.make('Reacher4Dof-v0')
# env = gym.make('Reacher5Dof-v0')
env = gym.make('Reacher6Dof-v0')

# print("starting check")
# check_env(env, warn=True)
# print("check done")

env.render(mode="human")   

print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for episode in range(50):
    obs = env.reset()             
    rewards = []
    
    for t in range(100):
        action = env.action_space.sample()  
        obs, reward, done, info = env.step(action) 
        # print ("Episode: {0}\n Time step: {1}\n Action: {2}\n Observation: {3}\n Reward: {4}\n done: {5} \n info: {6}".format(episode, t, action, obs, reward, done, info))
        rewards.append(reward)
        time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  
    
env.close()


