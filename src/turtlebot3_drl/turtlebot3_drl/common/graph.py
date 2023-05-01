import numpy
import os

import matplotlib.pyplot as plt
from turtlebot3_drl.drl_environment.reward import SUCCESS
from .settings import MODEL_STORE_INTERVAL

class Graph():
    def __init__(self):
        plt.ion()
        plt.show()

        self.session_dir = ""
        self.bin_size_average_reward = 3
        self.legend_labels = ['Unknown', 'Success', 'Collision Wall', 'Collision Dynamic', 'Timeout', 'Tumble']
        self.legend_colors = ['b', 'g', 'r', 'c', 'm', 'y']

        self.outcome_histories = []

        self.global_steps = 0
        self.data_outcome_history = []
        self.data_rewards = []
        self.data_loss_critic = []
        self.data_loss_actor = []
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.set_size_inches(18.5, 10.5)

        titles = ['outcomes', 'avg critic loss over episode', 'avg actor loss over episode', 'avg reward over 10 episodes']
        for i in range(4):
            ax = self.ax[int(i/2)][int(i%2!=0)]
            ax.set_title(titles[i])
        self.legend_set = False

    def set_graphdata(self, graphdata, episode):
        self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor = [graphdata[i] for i in range(len(self.graphdata))]
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]
        self.draw_plots(episode)
        return self.global_steps

    def update_data(self, step, global_steps, outcome, reward_sum, loss_critic_sum, loss_actor_sum):
        self.global_steps = global_steps
        self.data_outcome_history.append(outcome)
        self.data_rewards.append(reward_sum)
        self.data_loss_critic.append(loss_critic_sum / step)
        self.data_loss_actor.append(loss_actor_sum / step)
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

    def draw_plots(self, episode):
        xaxis = numpy.array(range(episode))

        # Plot outcome history
        for idx in range(len(self.data_outcome_history)):
            if idx == 0:
                self.outcome_histories = [[0],[0],[0],[0],[0],[0]]
                self.outcome_histories[self.data_outcome_history[0]][0] += 1
            else:
                for outcome_history in self.outcome_histories:
                    outcome_history.append(outcome_history[-1])
                foo = self.outcome_histories[self.data_outcome_history[idx]]
                foo[-1] += 1

        if len(self.data_outcome_history) > 0:
            i = 0
            for outcome_history in self.outcome_histories:
                self.ax[0][0].plot(xaxis, outcome_history, color=self.legend_colors[i], label=self.legend_labels[i])
                i += 1
            if not self.legend_set:
                self.ax[0][0].legend()
                self.legend_set = True

        # Plot critic loss
        y = numpy.array(self.data_loss_critic)
        self.ax[0][1].plot(xaxis, y)

        # Plot actor loss
        y = numpy.array(self.data_loss_actor)
        self.ax[1][0].plot(xaxis, y)

        # Plot average reward
        count = int(episode / self.bin_size_average_reward)
        if count > 0:
            xaxis = numpy.array(range(self.bin_size_average_reward, episode+1, self.bin_size_average_reward))
            averages = list()
            for i in range(count):
                avg_sum = 0
                for j in range(self.bin_size_average_reward):
                    avg_sum += self.data_rewards[i * self.bin_size_average_reward + j]
                averages.append(avg_sum / self.bin_size_average_reward)
            y = numpy.array(averages)
            self.ax[1][1].plot(xaxis, y)

        plt.draw()
        plt.pause(0.05)
        plt.savefig(os.path.join(self.session_dir, "_figure.png"))

    def get_success_count(self):
        suc = self.data_outcome_history[-MODEL_STORE_INTERVAL:]
        return suc.count(SUCCESS)

    def get_reward_average(self):
        rew = self.data_rewards[-MODEL_STORE_INTERVAL:]
        return sum(rew) / len(rew)
