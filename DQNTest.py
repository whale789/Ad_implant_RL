# # from collections import namedtuple
# #
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import numpy as np
# # import gym
# #
# #
# # # 定义经验回放缓冲区的经验元组
# # Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
# #
# # # 定义神经网络模型
# # class DQNModel(torch.nn.Module):
# #     def __init__(self,state_size,action_size):
# #         print(state_size)
# # # 定义DQN模型
# # class DQN:
# #       def __init__(self,state_size,action_size):
# #           # 初始化神经网络和目标网络
# #           print(self)
# #           self.q_network = DQNModel(state_size, action_size)
# #           self.target_network = DQNModel(state_size, action_size)
# #           # 将目标网络的权重设置为与 Q 网络相同
# #           self.target_network.load_state_dict(self.q_network.state_dict())
# #           # 其他初始化操作
# #
# #       def load_state_dict(self, param):
# #           print("这是加载模型字典")
# #
# #       def state_dict(self):
# #           pass
# #
# #
# # # 定义经验回放缓冲区
# # class ReplayBuffer:
# #       def __init__(self):
# #           print("这是经验缓冲区")
# #
# # # 定义DQN代理
# # class DQNAgent:
# #     def __init__(self,state_size,action_size):
# #         self.model = None
# #         self.state_size=state_size
# #         self.action_size=action_size
# #     def select_action(self,state):
# #         print("这是动作选择,state为：",state)
# #         return state
# #
# #     def train(self, batch_size):
# #         print("这是训练，batch_size为：",batch_size)
# #
# #     def update_target_model(self):
# #         print("这是更新目标模块")
# #
# #
# # # 创建环境
# # env = gym.make('CartPole-v1')
# # state_size = env.observation_space.shape[0]
# # action_size = env.action_space.n
# # # print(state_size,action_size)
# # # 创建DQN模型和目标网络
# # dqn_model = DQN(state_size, action_size)
# # target_model = DQN(state_size, action_size)
# # target_model.load_state_dict(dqn_model.state_dict())
# #
# # # 创建经验回放缓冲区和DQN代理
# # replay_buffer = ReplayBuffer(capacity=10000)
# # agent = DQNAgent(state_size, action_size)
# #
# # # 训练循环
# # num_episodes = 1000
# # batch_size = 32
# #
# # for episode in range(num_episodes):
# #     state = env.reset()
# #     total_reward = 0
# #
# #     while True:
# #         action = agent.select_action(state)
# #         next_state, reward, done, _ = env.step(action)
# #         replay_buffer.push((state, action, reward, next_state, done))
# #         state = next_state
# #
# #         agent.train(batch_size)
# #
# #         total_reward += reward
# #
# #         if done:
# #             agent.update_target_model()
# #             break
# #
# #     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
# #
# # # 保存训练好的模型
# # torch.save(agent.model.state_dict(), 'dqn_model.pth')
# #
# # # 测试代码
# # # ...
# #
# # # 关闭环境
# # env.close()
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
# import collections
#
# class Net(nn.Module):
#     def __init__(self, n_states, n_hidden, n_actions):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(n_states, n_hidden)
#         self.fc2 = nn.Linear(n_hidden, n_actions)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# class DQN(object):
#     def __init__(self, n_states, n_actions, n_hidden, lr, gamma, epsilon, memory_capacity, batch_size):
#         self.eval_net, self.target_net = Net(n_states, n_hidden, n_actions), Net(n_states, n_hidden, n_actions)
#         self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
#         self.loss_func = nn.MSELoss()
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.n_hidden = n_hidden
#         self.lr = lr
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.memory_capacity = memory_capacity
#         self.batch_size = batch_size
#         self.memory_counter = 0
#         self.learn_step_counter = 0
#
#     def choose_action(self, state):
#         state = torch.unsqueeze(torch.FloatTensor(state), 0)
#         if np.random.uniform() < self.epsilon:
#             actions_value = self.eval_net.forward(state)
#             action = torch.max(actions_value, 1)[1].data.numpy()[0]
#         else:
#             action = np.random.randint(0, self.n_actions)
#         return action
#
#     def store_transition(self, state, action, reward, next_state):
#         transition = np.hstack((state, [action, reward], next_state))
#         index = self.memory_counter % self.memory_capacity
#         self.memory[index, :] = transition
#         self.memory_counter += 1
#
#     def learn(self):
#         if self.learn_step_counter % 100 == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#         self.learn_step_counter += 1
#         sample_index = np.random.choice(self.memory_capacity, self.batch_size)
#         b_memory = self.memory[sample_index, :]
#         b_state = torch.FloatTensor(b_memory[:, :self.n_states])
#         b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
#         b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
#         b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])
#         q_eval = self.eval_net(b_state).gather(1, b_action)
#         q_next = self.target_net(b_next_state).detach()
#         q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
#         loss = self.loss_func(q_eval, q_target)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
from random import random

class RTB:
    def __init__(self,ad_state_x,ad_state_y):
        self.ad_x=ad_state_x[0]
        # self.ad_location_x=self.ad_x['ad_location_x']
        self.ad_y=ad_state_y[0]
        # self.ad_location_y=self.ad_y['ad_location_y']
        print(self.ad_x,self.ad_y)
import random
ad_width=0.3
ad_heigth=0.2

ad_state_x=[random.uniform(ad_width/2,1-(ad_width/2)) for _ in range(5)]
ad_state_y=[random.uniform(ad_heigth/2,1-(ad_heigth/2)) for _ in range(5)]

rt=RTB(ad_state_x,ad_state_y)
print(ad_state_x,ad_state_y)
