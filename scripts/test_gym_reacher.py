import gym
import gym_reacher
import time

# env = gym.make('Reacher1Dof-v0')
env = gym.make('Reacher2Dof-v0')
# env = gym.make('Reacher3Dof-v0')
# env = gym.make('Reacher4Dof-v0')
# env = gym.make('Reacher5Dof-v0')
# env = gym.make('Reacher6Dof-v0')
env.render(mode="human")   

print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


for episode in range(20):
    state = env.reset()             
    rewards = []
    
    for t in range(100):
        action = env.action_space.sample()  
        state, reward, done, info = env.step(action) 

        print ("Episode: {0}\n Time step: {1}\n Action: {2}\n State: {3}\n Reward: {4}".format(episode, t, action, state, reward))
        rewards.append(reward)
        time.sleep(1./30.) 

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(episode, cumulative_reward))  
    
env.close()


