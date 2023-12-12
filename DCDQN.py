import os
import random

import numpy as np
import torch.nn
import torch.nn.functional as F

from env import Ad_Environment
import matplotlib.pyplot as plt

N_Actions=5  #动作数
N_States=2  #状态数
Memory_Capacity=10 #记忆库容量
Batch_size=5  #样本数量
LR=0.01 #学习率
Epsilon=0.9  #贪心策略
Target_Replace_iter=10 #目标网络更新频率
Gamma=0.9  #奖励折扣

class DQNNet(torch.nn.Module):  #定义网络
    def __init__(self):
        super(DQNNet,self).__init__()
        self.fc1=torch.nn.Linear(N_States,5)  #建立第一个全连接层，状态个数神经元到50个神经元
        self.fc1.weight.data.normal_(0,0.1)   #权重初始化，均值为0，方差为0.1的正态分布
        self.out=torch.nn.Linear(5,N_Actions) #建立第二个全连接层，50个神经元到动作个数神经元
        self.out.weight.data.normal_(0,0.1)   #权重初始化，均值为0，方差为0.1的正态分布
    def forward(self,x):   #x为状态
        x=F.relu(self.fc1(x))  #连接输入层到隐藏层，且使用激励函数Relu函数来处理经过隐藏层后的值
        actions_value=self.out(x)  #连接隐藏层到输出层，获得最终的输出值，即动作值
        # print("111",actions_value)
        return actions_value    #返回动作值

class DQN(object):
    def __init__(self):    #定义DQN的一系列属性
        self.eval_net,self.target_net=DQNNet(),DQNNet()  #利用DQNNet创建两个网络：评估网络和目标网路
        self.learn_step_counter=0     #for target updating 目标网络的更新
        self.memory_counter=0   #for storing memory 记忆库储存
        self.memory=np.zeros((Memory_Capacity,N_States*2+2))  #初始化记忆库，一行代表一个transition（过渡）
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR) #使用Adam优化器，输入为评估网路的参数和优化器
        self.loss_func=torch.nn.MSELoss()   #使用均方损失函数，loss(xi,yi)=(xi-yi)^2

    def choose_action(self,state):   #定义动作选择函数，x为状态
        x=torch.unsqueeze(torch.FloatTensor(state),0)  #将x转换为floating point的形式，并在dim=0增加维数为1的维度
        if np.random.uniform()<Epsilon:    #生成一个[0,1]之内的随机数，如果小于Epslion，选择最优动作
            actions_value=self.eval_net.forward(x)  #通过评估网络输入状态x，前向传播获取动作值
            action=torch.max(actions_value,1)[1].data.numpy()   #输出每一行的最大值的索引，并转换为numpy ndarray形式
            action=action[0]  #输出action的第一个数
            # print("333",action)
        else:  #随机选择动作
            action=np.random.randint(0,N_Actions)  #随机选择动作总数之间的一个动作
        return action

    def store_transition(self,s,a,r,s_):   #定义记忆储存函数（这里的输入为一个transition）
        transition=np.hstack((s,[a,r],s_))   #在水平方向拼接数组
        # print("123",transition)
        #如果记忆库满了，便覆盖旧的数据
        index=self.memory_counter%Memory_Capacity   #获取transition要置入的行数
        self.memory[index,:]=transition   #置入transition
        self.memory_counter+=1

    def learn(self):  #定义学习函数，记忆库满后开始学习
        #目标网络参数更新
        if self.learn_step_counter%Target_Replace_iter==0:    #一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  #将评估网路的参数赋值给目标网络
        self.learn_step_counter+=1


        sample_index=np.random.choice(Memory_Capacity,Batch_size)   #在[0,Memory_Capacity]中随机抽取Batch_size个数，可能会重复
        b_memory=self.memory[sample_index,:]   #抽取sample_index个索引对应的sample_index个transition存到b_memory
        #将32个s抽出，并转化成32-bit floating point形式，并储存到b_s中，b_s大小为32行4列
        b_s=torch.FloatTensor(b_memory[:,:N_States])
        #将32个a抽出，并转化为64-bit integer(signed)形式，并储存到b_a中，b_a大小为32行1列
        #之所以改成LongTensor使用，是为了方便后面torch.gather使用
        b_a=torch.LongTensor(b_memory[:,N_States:N_States+1].astype(int))
        #将32个r抽出，并转化成32-bit floating point形式，并储存到b_r中，b_r大小为32行1列
        b_r=torch.FloatTensor(b_memory[:,N_States+1:N_States+2])
        #将32个s_抽出，并转换为32-bit floating point形式，并储存到b_s_中，b_s_大小为32列4行
        b_s_=torch.FloatTensor(b_memory[:,-N_States:])

        #获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        #eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1,b_a)代表对每行索引b_a对应的Q值进行聚合
        q_eval=self.eval_net(b_s).gather(1,b_a)
        #q_next表示不进行反向传播，所以detach表示通过目标网络输出32行b_s_对应的一系列动作值
        q_next=self.target_net(b_s_).detach()
        #q_next.max(1)[0]表示只返回每一行的最大值，不反回索引（长度为32的一个张量）.view表示把之前获得的一维张量变成（Batch_size,1）的形状，最终通过公式获得目标值
        q_target=b_r+Gamma*q_next.max(1)[0].view(Batch_size,1)
        #输入32个评估值和32个目标值，使用均方损失函数
        loss=self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()   #清空上一步的残余更新参数值
        loss.backward()  #误差反向传播，计算参数更新值
        self.optimizer.step()   #更新评估网络的所有参数


def read_file():
    folder_path="Datas/Gaze_txt_files"
    destination_folder_path = "Datas/Gaze_txt_files_Scenes"
    contents=os.listdir(folder_path)
    for content in contents:
        content_path=os.path.join(folder_path,content)
        if os.path.isdir(content_path):
            sub_contents=os.listdir(content_path)
            index=0
            for sub_content in sub_contents:
                sub_content_path=os.path.join(content_path,sub_content)

                print(index)
                index+=1
                # if os.path.isfile(sub_content_path):
                #     with open(sub_content_path,'r',newline='') as file:
                #         file_content=file.readlines()


def main():
    episodes=200
    dqn=DQN()

    ad_counter=5  #广告候选空间数量

    #2023.12.2
    #以下为随机生成的0到1之间的(x,y)坐标，模拟植入广告的中心点，生成5组(x,y)坐标，以及手动固定其width和heigth，因此只考虑平移情况

    ad_width=0.2
    ad_heigth=0.2


    # ad_state_x=[random.uniform(0.3+ad_width/2,0.8-(ad_heigth/2)) for _ in range(ad_counter)]
    # ad_state_y=[random.uniform(0.2+ad_heigth/2,0.7-(ad_heigth/2)) for _ in range(ad_counter)]
    # ad_state_x=random.uniform(0.3+ad_width/2,0.8-(ad_heigth/2))
    # ad_state_y=random.uniform(0.2+ad_heigth/2,0.7-(ad_heigth/2))
    ad_state_x=4.5
    ad_state_y=5.0
    layer=random.randint(0,ad_counter-1)
    # print(layer)
    ad_limit_x=5.0
    ad_limit_y=4.0
    ad_limit_width=4.0
    ad_limit_height=3.0
    env=Ad_Environment(ad_state_x,ad_state_y,layer,ad_counter,ad_width,ad_heigth,ad_limit_x,ad_limit_y,ad_limit_width,ad_limit_height,
                       total_step=100,ad_density=0)
    max_reward=float('-inf')
    for i in range(episodes):
        print('Episodes:%s' %i)
        s=env.reset()   #重置环境
        # print('123',s)
        episode_reward_sum=0

        while True:
            a=dqn.choose_action(s)
            s_,r,done=env.step(a)
            dqn.store_transition(s,a,r,s_)   #储存样本到数据库中
            episode_reward_sum+=r  #逐步加上一个episodes内的每个step的reward

            s=s_ #更新状态

            if dqn.memory_counter>Memory_Capacity:
                #开始学习（抽取记忆，即32个transition,对评估网络的参数进行更新，并且每个100次讲评估网络的参数赋给目标网络）
                dqn.learn()
            if done:
                print("episode:%s---episode-reward:%s" %(i,episode_reward_sum))
                break
        # print("111",episode_reward_sum)
        model=DQNNet()
        if episode_reward_sum>max_reward:
            max_reward=episode_reward_sum
            torch.save(model.state_dict(),"best_model.pth")

def test():
    ad_counter = 5  # 广告候选空间数量
    dqn = DQN()
    model=DQNNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    ad_width = 0.2
    ad_heigth = 0.2
    ad_state_x = [random.uniform(0.3 + ad_width / 2, 0.8 - (ad_heigth / 2)) for _ in range(ad_counter)]
    ad_state_y = [random.uniform(0.2 + ad_heigth / 2, 0.7 - (ad_heigth / 2)) for _ in range(ad_counter)]

    layer = random.randint(0, ad_counter - 1)
    # print(layer)
    env = Ad_Environment(ad_state_x, ad_state_y, layer, ad_counter, ad_width, ad_height=ad_heigth, total_step=100,
                         ad_density=0)
    s=env.reset()
    x = torch.unsqueeze(torch.FloatTensor(s), 0)
    with torch.no_grad():
        action_probabilities=model(x)
    choose_action=torch.argmax(action_probabilities).item()
    print(choose_action)


if __name__ == "__main__":
    # main()
    # test()
    read_file()