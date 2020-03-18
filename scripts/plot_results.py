from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np




def create_df(env_name):

    df1 = pd.read_csv('../results/'+env_name+'/time/train_time.csv')
    df2 = pd.read_csv('../results/'+env_name+'/accuracy/accuracy_reach.csv')

    df1['train time (ms)'] = df1['train_time (ms)']/(1000*60)
    df1 = df1.drop('train_time (ms)', axis=1)

    df1['success ratio'] = df2['success ratio']
    df1['average reach time'] = df2['average reach time']

    df1.rename({'algo': 'Algorithm', 'train time (ms)': 'train time (min)', 'success ratio': 'success ratio', 'average reach time': 'average reach time'}, axis=1, inplace=True)

    return df1

df_reach1 = create_df('Reacher1Dof-v0')
df_reach2 = create_df('Reacher2Dof-v0')
df_reach3 = create_df('Reacher3Dof-v0')
df_reach4 = create_df('Reacher4Dof-v0')
df_reach5 = create_df('Reacher5Dof-v0')
df_reach6 = create_df('Reacher6Dof-v0')

print(df_reach1)
print(df_reach2)
print(df_reach3)
print(df_reach4)
print(df_reach5)
print(df_reach6)


def plot_per_robot(df_reach, filename, titlename):


    ax = df_reach.plot(
        x="Algorithm", 
        y=["train time (min)", "success ratio", "average reach time"], 
        kind="bar", 
        rot=0, 
        subplots=True, 
        layout=(1, 3),
        sharex=False, 
        title=['', '', ''], 
        legend=None,
        figsize=(15, 5)
        )

    # print(ax.shape)
    ax[0, 0].set_ylabel("train time (min)")
    ax[0, 1].set_ylabel("success ratio")
    ax[0, 2].set_ylabel("average reach time")

    ax[0, 0].set_ylim(0, 50)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 2].set_ylim(0, 90)

    ax[0, 1].set_title(titlename)

    ax[0, 0].tick_params(axis='both', which='both', labelsize=8)
    ax[0, 1].tick_params(axis='both', which='both', labelsize=8)
    ax[0, 2].tick_params(axis='both', which='both', labelsize=8)

    # plt.show()
    plt.savefig("../results/"+filename, dpi=300, bbox_inches='tight')


plot_per_robot(df_reach1, "Reacher1Dof-v0/reacher1.pdf", "Reacher - 1 DoF")
plot_per_robot(df_reach2, "Reacher2Dof-v0/reacher2.pdf", "Reacher - 2 DoF")
plot_per_robot(df_reach3, "Reacher3Dof-v0/reacher3.pdf", "Reacher - 3 DoF")
plot_per_robot(df_reach4, "Reacher4Dof-v0/reacher4.pdf", "Reacher - 4 DoF")
plot_per_robot(df_reach5, "Reacher5Dof-v0/reacher5.pdf", "Reacher - 5 DoF")
plot_per_robot(df_reach6, "Reacher6Dof-v0/reacher6.pdf", "Reacher - 6 DoF")



df_reach1['Robot'] = 'Reacher 1DoF'
df_reach2['Robot'] = 'Reacher 2DoF'
df_reach3['Robot'] = 'Reacher 3DoF'
df_reach4['Robot'] = 'Reacher 4DoF'
df_reach5['Robot'] = 'Reacher 5DoF'
df_reach6['Robot'] = 'Reacher 6DoF'

# print(df_reach2)
df = pd.concat([df_reach1, df_reach2, df_reach3, df_reach4, df_reach5, df_reach6], ignore_index=True)

print(df)
df.to_csv("../results/res_file.csv", index=False, float_format='%.2f') 

df_A2C = df.loc[df['Algorithm'] == "A2C"]
df_ACKTR = df.loc[df['Algorithm'] == "ACKTR"]
df_DDPG = df.loc[df['Algorithm'] == "DDPG"]
df_PPO1 = df.loc[df['Algorithm'] == "PPO1"]
df_PPO2 = df.loc[df['Algorithm'] == "PPO2"]
df_SAC = df.loc[df['Algorithm'] == "SAC"]
df_TRPO = df.loc[df['Algorithm'] == "TRPO"]
df_TD3 = df.loc[df['Algorithm'] == "TD3"]

print(df_A2C)


def plot_per_algo(df_algo, filename, titlename):

    ax = df_algo.plot(
            x="Robot", 
            y=["train time (min)", "success ratio", "average reach time"], 
            kind="bar", 
            rot=45, 
            subplots=True, 
            layout=(1, 3),
            sharex=False, 
            title=['', '', ''], 
            legend=None,
            figsize=(15, 5)
            )

    # print(ax.shape)
    ax[0, 0].set_ylabel("train time (min)")
    ax[0, 1].set_ylabel("success ratio")
    ax[0, 2].set_ylabel("average reach time")

    ax[0, 0].set_ylim(0, 50)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 2].set_ylim(0, 90)

    ax[0, 1].set_title(titlename)

    ax[0, 0].tick_params(axis='both', which='both', labelsize=8)
    ax[0, 1].tick_params(axis='both', which='both', labelsize=8)
    ax[0, 2].tick_params(axis='both', which='both', labelsize=8)

    # plt.show()
    plt.savefig("../results/"+filename, dpi=300, bbox_inches='tight')


plot_per_algo(df_A2C, "A2C.pdf", "A2C")
plot_per_algo(df_ACKTR, "ACKTR.pdf", "ACKTR")
plot_per_algo(df_DDPG, "DDPG.pdf", "DDPG")
plot_per_algo(df_PPO1, "PPO1.pdf", "PPO1")
plot_per_algo(df_PPO2, "PPO2.pdf", "PPO2")
plot_per_algo(df_SAC, "SAC.pdf", "SAC")
plot_per_algo(df_TD3, "TD3.pdf", "TD3")
plot_per_algo(df_TRPO, "TRPO.pdf", "TRPO")


print(df.groupby(['Robot']).mean())
print(df.groupby(['Algorithm']).mean())
