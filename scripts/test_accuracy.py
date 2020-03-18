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

episode_nb = 0
successful_episodes = 0

while episode_nb <= 10:

    state = env.reset()             
    rewards = []
    timestep_successful_ep = []
    
    # only run an episode if the target is reachable
    robot_reach, dist_target_origin, dist_ft_t = env.extra_info()
    # print("distance origin to target:", dist_target_origin)
    # print("robot reach:", robot_reach)
    # print("distance fingertip - target:", dist_ft_t)
    
    if dist_target_origin <= robot_reach:
        print("episode: ", episode_nb)

        for t in range(100):
            action = env.action_space.sample()  
            state, reward, done, info = env.step(action) 
            # print ("Episode: {0}\n Time step: {1}\n Action: {2}\n State: {3}\n Reward: {4}".format(episode_nb, t, action, state, reward))
            rewards.append(reward)

            time.sleep(1./30.) 

            cumulative_reward = sum(rewards)
        print("episode {} | cumulative reward : {}".format(episode_nb, cumulative_reward))  
    
        if done:
            successful_episodes += 1
            timestep_successful_ep.append(t)
            break

        episode_nb += 1
    
    else:
        print("Target not reachable, a new target will be generated.")
    
success_ratio = successful_episodes/episode_nb
print("sucess ratio:", success_ratio)

if successful_episodes != 0:
    average_reach_time = sum(timestep_successful_ep)/successful_episodes
    print("average_reach_time:", average_reach_time)

env.close()


