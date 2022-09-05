import os
import time
import numpy as np
import pandas as pd
import torch
from math import floor, ceil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from job_img3_re import JobEnv


def SpatialPyramidPooling2d(input_x, level, pool_type='max_pool'):
    N, C, H, W = input_x.size()
    for i in range(level):
        level = i + 1
        kernel_size = (ceil(H / level), ceil(W / level))
        stride = (ceil(H / level), ceil(W / level))
        padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))

        if pool_type == 'max_pool':
            tensor = (F.max_pool2d(input_x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
        else:
            tensor = (F.avg_pool2d(input_x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)

        if i == 0:
            res = tensor
        else:
            res = torch.cat((res, tensor), 1)
    return res


def _cal_num_grids(level):
    count = 0
    for i in range(level):
        count += (i + 1) * (i + 1)
    return count


class Actor_SPP(nn.Module):
    def __init__(self, input_channel=3, out_channel=16, out_num=6, num_level=4):
        super(Actor_SPP, self).__init__()
        self.num_level = num_level
        self.num_grid = _cal_num_grids(num_level)
        self.feature1 = nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=(5, 5), padding=2), nn.ReLU())
        self.linear1 = nn.Linear(out_channel * self.num_grid, out_num)

    def forward(self, x):
        x = self.feature1(x)
        x = SpatialPyramidPooling2d(x, self.num_level)
        action_prob = F.softmax(self.linear1(x), dim=1)
        return action_prob


class Critic_SPP(nn.Module):
    def __init__(self, input_channel=3, out_channel=16, out_num=1, num_level=4):
        super(Critic_SPP, self).__init__()
        self.num_level = num_level
        self.num_grid = _cal_num_grids(num_level)
        self.feature1 = nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=(5, 5), padding=2), nn.ReLU())
        self.linear1 = nn.Linear(out_channel * self.num_grid, out_num)

    def forward(self, x):
        x = self.feature1(x)
        x = SpatialPyramidPooling2d(x, self.num_level)
        state_value = self.linear1(x)
        return state_value


class PPO:
    def __init__(self, j_env, memory_size=5, batch_size=32, clip_ep=0.2):
        super(PPO, self).__init__()
        self.env = j_env
        self.memory_size = memory_size
        self.batch_size = batch_size  # update batch size
        self.epsilon = clip_ep

        self.action_dim = self.env.action_num
        self.case_name = self.env.case_name
        self.gamma = 1  # reward discount
        self.A_LR = 1e-3  # learning rate for actor
        self.C_LR = 3e-3  # learning rate for critic
        self.A_UPDATE_STEPS = 10  # actor update steps
        self.max_grad_norm = 0.5
        self.training_step = 0

        self.actor_net = Actor_SPP(out_num=self.action_dim)
        self.critic_net = Critic_SPP(out_num=1)
        self.actor_optimizer = optimizer.Adam(self.actor_net.parameters(), self.A_LR)
        self.critic_net_optimizer = optimizer.Adam(self.critic_net.parameters(), self.C_LR)

        if not os.path.exists('param'):
            os.makedirs('param')

    def select_action(self, state):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float)
        state = state_tensor.unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_params(self, instance_name):
        torch.save(self.actor_net.state_dict(), 'param/' + instance_name + '_actor_net.model')
        torch.save(self.critic_net.state_dict(), 'param/' + instance_name + '_critic_net.model')

    def load_params(self, instance_name):
        self.critic_net.load_state_dict(torch.load('param/' + instance_name + '_critic_net.model'))
        self.actor_net.load_state_dict(torch.load('param/' + instance_name + '_actor_net.model'))

    def update(self, bs, ba, br, bp):
        # get old actor log prob
        old_action_log_prob = torch.tensor(bp, dtype=torch.float).view(-1, 1)
        state = torch.tensor(np.array(bs), dtype=torch.float)
        action = torch.tensor(ba, dtype=torch.long).view(-1, 1)
        d_reward = torch.tensor(br, dtype=torch.float)

        for i in range(self.A_UPDATE_STEPS):
            for index in BatchSampler(SubsetRandomSampler(range(len(ba))), self.batch_size, False):
                new_state = state[index]
                #  compute the advantage
                d_reward_index = d_reward[index].view(-1, 1)
                V = self.critic_net(new_state)
                delta = d_reward_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!
                action_prob = self.actor_net(new_state).gather(1, action[index])  # new policy
                ratio = (action_prob / old_action_log_prob[index])
                surrogate = ratio * advantage
                clip_loss = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                action_loss = -torch.min(surrogate, clip_loss).mean()

                # update actor network
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(d_reward_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

    def train(self, data_set, save_params=False):
        if not save_params:
            self.load_params(data_set)
        column = ["episode", "make_span", "reward"]
        results = pd.DataFrame(columns=column, dtype=float)
        index = 0
        converged = 0
        converged_value = []
        t0 = time.time()
        for i_epoch in range(4000):
            if time.time() - t0 >= 3600:
                break
            bs, ba, br, bp = [], [], [], []
            for m in range(self.memory_size):  # memory size is the number of complete episode
                buffer_s, buffer_a, buffer_r, buffer_p = [], [], [], []
                state = self.env.reset()
                episode_reward = 0
                while True:
                    action, action_prob = self.select_action(state)
                    next_state, reward, done = self.env.step(action)
                    buffer_s.append(state)
                    buffer_a.append(action)
                    buffer_r.append(reward)
                    buffer_p.append(action_prob)

                    state = next_state
                    episode_reward += reward
                    if done:
                        v_s_ = 0
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + self.gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()

                        bs[len(bs):len(bs)] = buffer_s
                        ba[len(ba):len(ba)] = buffer_a
                        br[len(br):len(br)] = discounted_r
                        bp[len(bp):len(bp)] = buffer_p

                        # Episode: make_span: Episode reward
                        print('{}    {}    {:.2f} {}'.format(i_epoch, self.env.current_time, episode_reward,
                                                             self.env.no_op_cnt))
                        index = i_epoch * self.memory_size + m
                        results.loc[index] = [i_epoch, self.env.current_time, episode_reward]
                        converged_value.append(self.env.current_time)
                        if len(converged_value) >= 31:
                            converged_value.pop(0)
                        break
            self.update(bs, ba, br, bp)
            converged = index
            if min(converged_value) == max(converged_value) and len(converged_value) >= 30:
                converged = index
                break
        if not os.path.exists('results'):
            os.makedirs('results')
        results.to_csv("results/" + str(self.env.case_name) + "_" + data_set + ".csv")
        if save_params:
            self.save_params(data_set)
        return min(converged_value), converged, time.time() - t0, self.env.no_op_cnt

    def test(self, data_set):
        self.load_params(data_set)
        state = self.env.reset()
        while True:
            action, _ = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            state = next_state
            if done:
                break
        print(self.env.current_time)


if __name__ == '__main__':
    # training policy
    parameters = "data_set_rescheduling_new_middle"
    path = "../data_set_rescheduling_new_middle/"
    print(parameters)
    param = [parameters, "converge_cnt", "total_time", "no_op"]
    simple_results = pd.DataFrame(columns=param, dtype=int)
    for file_name in os.listdir(path):
        print(file_name + "========================")
        title = file_name.split('.')[0]
        name = file_name.split('_')[0]
        env = JobEnv(title, path)
        scale = env.job_num * env.machine_num
        model = PPO(env, memory_size=5, batch_size=2*scale, clip_ep=0.25)
        simple_results.loc[title] = model.train(title, save_params=True)
    simple_results.to_csv(parameters + "_result.csv")

    # path = "../all_data_set/"
    # env = JobEnv("la21", path)
    # scale = env.job_num * env.machine_num
    # model = PPO(env, memory_size=5, batch_size=2*scale, clip_ep=0.25)
    # # model.test(title)
    # model.train("la21")

