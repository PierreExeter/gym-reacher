import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import gym_reacher

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy


timesteps = 10000

# Create log dir
log_dir = "../results/tests/"
os.makedirs(log_dir, exist_ok=True)

# use wrappers
# env = gym.make('Pendulum-v0')
env = gym.make('Reacher3Dof-v0')

env = Monitor(env, filename=log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# train
model = A2C("MlpPolicy", env, verbose=1).learn(int(timesteps))

#plot results
results_plotter.plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "My title")
plt.show()

results_plotter.plot_results([log_dir], timesteps, results_plotter.X_EPISODES, "My title")
plt.show()

results_plotter.plot_results([log_dir], timesteps, results_plotter.X_WALLTIME, "My title")
plt.show()

print(load_results(log_dir))


#### hand-made plotting ####


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # x, y = ts2xy(load_results(log_folder), 'episodes')
    # x, y = ts2xy(load_results(log_folder), 'walltime_hrs')

    print(x)
    print(y)

    y = moving_average(y, window=10)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir)