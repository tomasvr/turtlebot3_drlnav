from cProfile import label
import matplotlib.pyplot as plt
import numpy
import os
import glob
import sys
import pandas as pd
import socket
from datetime import datetime

TOP_EPISODES = 4

def main(args=sys.argv[1:]):
    PLOT_INTERVAL = int(args[0])
    models = args[1:]

    plt.figure(figsize=(16,10))
    j = 0
    for model in models:
        base_path = os.getenv('DRLNAV_BASE_PATH') + "/src/turtlebot3_drl/model/" + str(socket.gethostname() + "/")
        if 'examples' in model:
            base_path = os.getenv('DRLNAV_BASE_PATH') + "/src/turtlebot3_drl/model/"
        logfile = glob.glob(base_path + model + '/_train_*.txt')
        if len(logfile) != 1:
            print(f"ERROR: found less or more than 1 logfile for: {base_path}{model}")
        df = pd.read_csv(logfile[0])
        rewards_column = df[' reward']
        rewards = rewards_column.tolist()
        average_rewards = []
        sum_rewards = 0
        episode_range = len(df.index)
        xaxis = numpy.array(range(0, episode_range - PLOT_INTERVAL, PLOT_INTERVAL))
        for i in range (episode_range):
            if i % PLOT_INTERVAL == 0 and i > 0:
                average_rewards.append(sum_rewards / PLOT_INTERVAL)
                sum_rewards = 0
            sum_rewards += rewards[i]
        plt.plot(xaxis, average_rewards, label=str(models[j]))
        top_episodes = list(numpy.argpartition(numpy.array(average_rewards), -TOP_EPISODES)[-TOP_EPISODES:])
        top_scores = list(numpy.array(average_rewards)[top_episodes])
        top_episodes = [i * PLOT_INTERVAL for i in top_episodes]
        print(f"model {model} best performing episodes: {top_episodes} with scores: {top_scores}")
        j += 1

    plt.xlabel('Episode', fontsize=24, fontweight='bold')
    plt.ylabel('Reward', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--')
    dt_string = datetime.now().strftime("%d-%m-%H:%M:%S")
    suffix = '-'.join(models).replace(' ', '_').replace('/', '-')
    plt.savefig(os.path.join(os.getenv('DRLNAV_BASE_PATH'), "util/graphs/", 'reward_graph_' + dt_string + '__' + suffix + ".png"), format="png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()