#!/usr/bin/env python3
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
# Authors: Tomas

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as torchf

from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING, ACTION_SIZE, HIDDEN_SIZE, BATCH_SIZE, BUFFER_SIZE, DISCOUNT_FACTOR, \
                                 LEARNING_RATE, TAU, STEP_TIME, EPSILON_DECAY, EPSILON_MINIMUM, STACK_DEPTH, FRAME_SKIP
from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES


class OffPolicyAgent(ABC):
    def __init__(self, device, simulation_speed):

        self.device = device
        self.simulation_speed   = simulation_speed

        # Network structure
        self.state_size         = NUM_SCAN_SAMPLES + 4
        self.action_size        = ACTION_SIZE
        self.hidden_size        = HIDDEN_SIZE
        self.input_size         = self.state_size
        # Hyperparameters
        self.batch_size         = BATCH_SIZE
        self.buffer_size        = BUFFER_SIZE
        self.discount_factor    = DISCOUNT_FACTOR
        self.learning_rate      = LEARNING_RATE
        self.tau                = TAU
        # Other parameters
        self.step_time          = STEP_TIME
        self.loss_function      = torchf.smooth_l1_loss
        self.epsilon            = 1.0
        self.epsilon_decay      = EPSILON_DECAY
        self.epsilon_minimum    = EPSILON_MINIMUM
        self.reward_function    = REWARD_FUNCTION
        self.backward_enabled   = ENABLE_BACKWARD
        self.stacking_enabled   = ENABLE_STACKING
        self.stack_depth        = STACK_DEPTH
        self.frame_skip         = FRAME_SKIP
        if ENABLE_STACKING:
            self.input_size *= self.stack_depth

        self.networks = []
        self.iteration = 0

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def get_action():
        pass

    @abstractmethod
    def get_action_random():
        pass

    def _train(self, replaybuffer):
        batch = replaybuffer.sample(self.batch_size)
        sample_s, sample_a, sample_r, sample_ns, sample_d = batch
        sample_s = torch.from_numpy(sample_s).to(self.device)
        sample_a = torch.from_numpy(sample_a).to(self.device)
        sample_r = torch.from_numpy(sample_r).to(self.device)
        sample_ns = torch.from_numpy(sample_ns).to(self.device)
        sample_d = torch.from_numpy(sample_d).to(self.device)
        result = self.train(sample_s, sample_a, sample_r, sample_ns, sample_d)
        self.iteration += 1
        if self.epsilon and self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay
        return result

    def create_network(self, type, name):
        network = type(name, self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.networks.append(network)
        return network

    def create_optimizer(self, network):
        return torch.optim.AdamW(network.parameters(), self.learning_rate)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_model_configuration(self):
        configuration = ""
        for attribute, value in self.__dict__.items():
            if attribute not in ['actor', 'actor_target', 'critic', 'critic_target']:
                configuration += f"{attribute} = {value}\n"
        return configuration

    def get_model_parameters(self):
        parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size,
                            self.discount_factor, self.learning_rate, self.tau, self.step_time, REWARD_FUNCTION,
                            ENABLE_BACKWARD, ENABLE_STACKING, self.stack_depth, self.frame_skip]
        parameter_string = ', '.join(map(str, parameters))
        return parameter_string

    def attach_visual(self, visual):
        self.actor.visual = visual

class Network(torch.nn.Module, ABC):
    def __init__(self, name, visual=None):
        super(Network, self).__init__()
        self.name = name
        self.visual = visual
        self.iteration = 0

    @abstractmethod
    def forward():
        pass

    def init_weights(n, m):
        if isinstance(m, torch.nn.Linear):
            # --- define weights initialization here (optional) ---
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)