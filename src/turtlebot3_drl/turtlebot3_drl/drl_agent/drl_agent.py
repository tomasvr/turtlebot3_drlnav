#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas

import copy
import os
import sys
import time
import numpy as np

from ..common.settings import ENABLE_VISUAL, ENABLE_STACKING, OBSERVE_STEPS, MODEL_STORE_INTERVAL

from ..common.storagemanager import StorageManager
from ..common.graph import Graph
from ..common.logger import Logger
if ENABLE_VISUAL:
    from ..common.visual import DrlVisual
from ..common import utilities as util

from .dqn import DQN
from .ddpg import DDPG
from .td3 import TD3

from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from ..common.replaybuffer import ReplayBuffer

class DrlAgent(Node):
    def __init__(self, algorithm, training, load_session="", load_episode=0, train_stage=util.test_stage):
        super().__init__(algorithm + '_agent')
        self.algorithm = algorithm
        self.is_training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        self.train_stage = train_stage
        if (not self.is_training and not self.load_session):
            quit("ERROR no test agent specified")
        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(self.train_stage)
        print(f"{'training' if (self.is_training) else 'testing' } on stage: {util.test_stage}")

        self.total_steps = 0
        self.observe_steps = OBSERVE_STEPS

        if self.algorithm == 'dqn':
            self.model = DQN(self.device, self.sim_speed)
        elif self.algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        elif self.algorithm == 'td3':
            self.model = TD3(self.device, self.sim_speed)
        else:
            quit(f"invalid algorithm specified: {self.algorithm}, chose one of: ddpg, td3, td3conv")

        self.replay_buffer = ReplayBuffer(self.model.buffer_size)
        self.graph = Graph()

        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        self.sm = StorageManager(self.algorithm, self.train_stage, self.load_session, self.episode, self.device)

        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            if self.is_training:
                self.replay_buffer.buffer = self.sm.load_replay_buffer(self.model.buffer_size, os.path.join(self.load_session, 'stage'+str(self.train_stage)+'_latest_buffer.pkl'))
            self.total_steps = self.graph.set_graphdata(self.sm.load_graphdata(), self.episode)
            print(f"global steps: {self.total_steps}")
            print(f"loaded model {self.load_session} (eps {self.episode}): {self.model.get_model_parameters()}")
        else:
            self.sm.new_session_dir()
            self.sm.store_model(self.model)

        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(self.is_training, self.sm.machine_dir, self.sm.session_dir, self.sm.session, self.model.get_model_parameters(), self.model.get_model_configuration(), str(util.test_stage), self.algorithm, self.episode)
        if ENABLE_VISUAL:
            self.visual = DrlVisual(self.model.state_size, self.model.hidden_size)
            self.model.attach_visual(self.visual)
        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #

        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        self.gazebo_pause = self.create_client(Empty, '/pause_physics')
        self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')
        self.process()


    def process(self):
        util.pause_simulation(self)
        while (True):
            episode_done = False
            step, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)

            if ENABLE_STACKING:
                frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
                state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)
                next_state = [0.0] * (self.model.state_size * self.model.stack_depth)

            util.unpause_simulation(self)
            time.sleep(0.5)
            episode_start = time.perf_counter()

            while not episode_done:
                if self.is_training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(state, self.is_training, step, ENABLE_VISUAL)

                action_current = action
                if self.algorithm == 'dqn':
                    action_current = self.model.possible_actions[action]

                # Take a step
                next_state, reward, episode_done, outcome, distance_traveled = util.step(self, action_current, action_past)
                action_past = copy.deepcopy(action_current)
                reward_sum += reward

                if ENABLE_STACKING:
                    frame_buffer = frame_buffer[self.model.state_size:] + list(next_state)      # Update big buffer with single step
                    next_state = []                                                         # Prepare next set of frames (state)
                    for depth in range(self.model.stack_depth):
                        start = self.model.state_size * (self.model.frame_skip - 1) + (self.model.state_size * self.model.frame_skip * depth)
                        next_state += frame_buffer[start : start + self.model.state_size]

                # Train
                if self.is_training == True:
                    self.replay_buffer.add_sample(state, action, [reward], next_state, [episode_done])
                    if self.replay_buffer.get_length() >= self.model.batch_size:
                        loss_c, loss_a, = self.model._train(self.replay_buffer)
                        loss_critic += loss_c
                        loss_actor += loss_a

                if ENABLE_VISUAL:
                    self.visual.update_reward(reward_sum)
                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            # Episode done
            util.pause_simulation(self)
            self.total_steps += step
            duration = time.perf_counter() - episode_start

            if self.total_steps >= self.observe_steps:
                self.episode += 1
                self.finish_episode(step, duration, outcome, distance_traveled, reward_sum, loss_critic, loss_actor)
            else:
                print(f"Observe steps completed: {self.total_steps}/{self.observe_steps}")

    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, lost_actor):
            print(f"Epi: {self.episode} R: {reward_sum:.2f} outcome: {util.translate_outcome(outcome)} \
                    steps: {step} steps_total: {self.total_steps}, time: {eps_duration:.2f}")
            if (self.is_training):
                self.graph.update_data(step, self.total_steps, outcome, reward_sum, loss_critic, lost_actor)
                self.logger.file_log.write(f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, {step}, {self.total_steps}, \
                                                {self.replay_buffer.get_length()}, {loss_critic / step}, {lost_actor / step}\n")

                if (self.episode % MODEL_STORE_INTERVAL == 0) or (self.episode == 1):
                    self.graph.draw_plots(self.episode)
                    self.sm.save_session(self.episode, self.model.networks, self.graph.graphdata, self.replay_buffer.buffer)
                    self.logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
            else:
                self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
                util.wait_new_goal(self)

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    drl_agent = DrlAgent(*args)
    rclpy.spin(drl_agent)
    drl_agent.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()