import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

import math
import numpy

from ..drl_environment.drl_environment import NUM_SCAN_SAMPLES
from turtlebot3_drl.drl_environment.reward import REWARD_FUNCTION
from ..common.settings import ENABLE_BACKWARD, ENABLE_STACKING

LINEAR = 0
ANGULAR = 1

class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, hidden_layer, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name
        self.iteration = 0

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fa1 = nn.Linear(state_size, hidden_layer)
        self.fa2 = nn.Linear(hidden_layer, hidden_layer)
        self.mean_linear = nn.Linear(hidden_layer, action_size)
        self.log_std_linear = nn.Linear(hidden_layer, action_size)

        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.mean_linear.weight)
        self.mean_linear.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.log_std_linear.weight)
        self.log_std_linear.bias.data.fill_(0.01)

    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

    def get_conv_sizes(self, input_size):
        #TODO: remove this
        return []

class Critic(nn.Module):

    def __init__(self, name, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # Q1
        self.l1 = nn.Linear(state_size, int(hidden_size / 2))
        self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.l1.weight)
        self.l1.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l2.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.l3.weight)
        self.l3.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.l4.weight)
        self.l4.bias.data.fill_(0.01)

        # Q2
        self.l5 = nn.Linear(state_size, int(hidden_size / 2))
        self.l6 = nn.Linear(action_size, int(hidden_size / 2))
        self.l7 = nn.Linear(hidden_size, hidden_size)
        self.l8 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.l5.weight)
        self.l5.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.l6.weight)
        self.l6.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.l7.weight)
        self.l7.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.l8.weight)
        self.l8.bias.data.fill_(0.01)


    def forward(self, states, actions):
        # q1_states = torch.relu(self.q1_fc1(states))
        # q1_merged = torch.cat((q1_states, actions), dim=1)
        # q1_x = torch.relu(self.q1_fca1(q1_merged))
        # q1_output = self.q1_fca2(q1_x)

        # q2_states = torch.relu(self.q2_fc1(states))
        # q2_merged = torch.cat((q2_states, actions), dim=1)
        # q2_x = torch.relu(self.q2_fca1(q2_merged))
        # q2_output = self.q2_fca2(q2_x)

        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        x1 = self.l4(x)

        xs = torch.relu(self.l5(states))
        xa = torch.relu(self.l6(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l7(x))
        x2 = self.l8(x)

        return x1, x2

        return q1_output, q2_output

class SAC():
    def __init__(self, device, sim_speed):
        self.device = device
        self.iteration = 0

        # DRL parameters
        self.batch_size      = 1024
        self.buffer_size     = 1000000
        self.state_size      = NUM_SCAN_SAMPLES + 4
        self.action_size     = 2
        self.hidden_size     = 512
        self.discount_factor = 0.99
        self.learning_rate   = 0.0001
        self.tau             = 0.0001
        self.step_time       = 0.0
        self.loss_function = F.smooth_l1_loss
        # SAC parameters
        self.log_std_min     = -20
        self.log_std_max     = 2
        self.alpha_start     = 0.5

        self.actor = Actor("actor", self.state_size, self.action_size, self.hidden_size, self.log_std_min, self.log_std_max).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_critic = Critic("target_critic", self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        self.hard_update(self.target_critic, self.critic)

        self.alpha = torch.tensor(self.alpha_start)
        self.target_entropy = -torch.prod(torch.Tensor([self.action_size]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], self.learning_rate)

        self.networks = [self.actor, self.critic, self.target_critic]

        self.parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size, self.discount_factor,
                                self.learning_rate, self.tau, self.step_time, self.actor_optimizer.__class__.__name__, self.critic_optimizer.__class__.__name__,
                                self.loss_function.__name__,  self.log_std_min, self.log_std_max, REWARD_FUNCTION, ENABLE_BACKWARD, sim_speed, ENABLE_STACKING,
                                self.alpha_start]


    def get_action(self, state, is_training, steps, visualize):
        #TODO: during training don't use sample but use mean action according to spinning up docs
        action, _, _, _ = self.actor.sample(torch.FloatTensor(state).to(self.device).unsqueeze(0))
        action = action.detach().cpu().data.numpy().tolist()[0]
        action[LINEAR] = numpy.clip(action[LINEAR], -1.0, 1.0)
        action[ANGULAR] = numpy.clip(action[ANGULAR], -1.0, 1.0)
        return action

    def train(self, replaybuffer):
        self.iteration += 1
        batch = replaybuffer.sample(self.batch_size)
        s_sample, a_sample, r_sample, new_s_sample, done_sample = batch

        s_sample = torch.from_numpy(s_sample).to(self.device)
        a_sample = torch.from_numpy(a_sample).to(self.device)
        r_sample = torch.from_numpy(r_sample).to(self.device).unsqueeze(1)
        new_s_sample = torch.from_numpy(new_s_sample).to(self.device)
        done_sample = torch.from_numpy(done_sample).to(self.device).unsqueeze(1)

        # optimize critic (q function)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.actor.sample(new_s_sample)
            target_q1, target_q2 = self.target_critic.forward(new_s_sample, next_state_action)
            target_Q = torch.minimum(target_q1, target_q2) - self.alpha * next_state_log_pi
            next_q_value = r_sample + (1 - done_sample) * self.discount_factor * (target_Q)

        current_q1, current_q2 = self.critic.forward(s_sample, a_sample)
        loss_critic = self.loss_function(current_q1, next_q_value) + self.loss_function(current_q2, next_q_value)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        # optimize actor (policy)
        pi, log_pi, mean, log_std = self.actor.sample(s_sample)

        # todo: use sum?
        # q1_loss_actor, q2_loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))
        qf1_pi, qf2_pi = self.critic.forward(s_sample, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

        # Regularization Loss
        #reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        #policy_loss += reg_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        self.soft_update(self.critic, self.target_critic, self.tau)
        return [loss_critic.detach().cpu(), actor_loss.detach().cpu()]
        return [loss_critic.detach(), actor_loss.detach(), alpha_loss.detach()]

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
