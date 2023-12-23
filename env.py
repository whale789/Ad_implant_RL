import copy
import os
import random

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class Ad_Environment:
    def __init__(self,ad_state_x,ad_state_y,layer,ad_counter,ad_width,ad_height,ad_limit_x,ad_limit_y,ad_limit_width,ad_limit_height,total_step,ad_density):
        self.layer=layer  #广告状态空间索引
        self.ad_counter=ad_counter-1   #广告空间总数
        self.ad_state_x=ad_state_x
        self.ad_state_y=ad_state_y
        self.density_layer=0
        self.total_reward=0
        # self.ad_location_x = ad_state_x[int(self.layer)]  # 广告水平位置
        # # print("00x",self.ad_location_x)
        # self.ad_location_y = ad_state_y[int(self.layer)]  # 广告垂直位置
        # print("00y",self.ad_location_y)
        self.ad_width=ad_width   #所植入广告的宽度
        self.ad_height=ad_height  #所植入广告的高度

        self.ad_limit_x=ad_limit_x  #限制区域中心点的x坐标
        self.ad_limit_y=ad_limit_y  #限制区域中心点的y坐标
        self.ad_limit_width=ad_limit_width  #限制区域的宽度
        self.ad_limit_height=ad_limit_height    #限制区域的高度

        self.total_step=total_step   ##总步数
        self.current_step=0  #当前步数
        self.ad_density=ad_density  #初始位置密度
        self.action_space=[0,1,2,3,4,5]  #分别代表up,down,left,right
        self.current_location_x=self.ad_state_x
        self.current_location_y=self.ad_state_y
        self.current_width=self.ad_width
        self.current_height=self.ad_height
    def step(self,action):
        if not action in self.action_space:
            print("该Action不存在")
        else:
            self.layer+=1
            if self.layer>self.ad_counter:
                self.layer=self.layer%self.ad_counter
            self.current_location_x = self.ad_state_x
            self.current_location_y = self.ad_state_y
            # print("222",self.current_location_x)

        if action==0:  #向上平移
           self.current_location_x=self.current_location_x
           self.current_location_y=self.current_location_y+0.05
        elif action==1:  #向下平移
            self.current_location_x=self.current_location_x
            self.current_location_y=self.current_location_y-0.05
        elif action==2:  #向左平移
            self.current_location_x=self.current_location_x-0.05
            self.current_location_y=self.current_location_y
        elif action==3:     #向右平移
            self.current_location_x=self.current_location_x+0.05
            self.current_location_y=self.current_location_y
        elif action==4:  #放大
            self.current_width=self.current_width+0.02
            self.current_height=self.current_height+0.02
        elif action==5:  #缩小
            self.current_width=self.current_width-0.02
            self.current_height=self.current_height-0.02
        # elif action==6:   #旋转
        #     # self.current_location_x-=1
        #     pass



        reward=self.calculate_reward()   #奖励函数

        self.current_step+=1
        # done=self.current_step>=self.total_step
        done=False
        if self.current_step>=self.total_step:
            done=True
            # print(self.density_layer)
            self.density_layer=(self.density_layer+1)%5
        return (self.current_location_x,self.current_location_y),reward,done

    def calculate_reward(self):
        #根据中心点和宽度、高度计算是否超出了限制区域
        # if self.current_location_x+self.current_width/2>self.ad_limit_x+self.ad_limit_width/2:
        #     self.total_reward-=((self.current_location_x+self.current_width/2)-(self.ad_limit_x+self.ad_limit_width/2))*10000
        # elif self.current_location_y+self.current_height/2>self.ad_limit_y+self.ad_limit_height/2:
        #     self.total_reward-=((self.current_location_y+self.current_height/2)-(self.ad_limit_y+self.ad_limit_height))*10000
        # elif self.current_location_x-self.current_width/2<self.ad_limit_x-self.ad_limit_width/2:
        #     self.total_reward+=((self.current_location_x-self.current_width/2)-(self.ad_limit_x-self.ad_limit_width/2))*10000
        # elif self.current_location_y-self.current_height/2<self.ad_limit_y-self.ad_limit_height/2:
        #     self.total_reward+=((self.current_location_y-self.current_height/2)-(self.ad_limit_y-self.ad_limit_height/2))*10000
        # else:
        density = self.area_density(self.current_location_x, self.current_location_y, self.ad_width,
                                    self.ad_height)  # 计算该区域的密度
        density_difference = density - self.ad_density
        self.total_reward = round(self.total_reward + round(density_difference, 4) / 100, 4)
        # print('333',density)
        self.ad_density = density

        return self.total_reward

    def get_state(self):   #讲ad_location_x和ad_location_y拼接起来
        cu_location_x=copy.copy(self.current_location_x)
        cu_location_y=copy.copy(self.current_location_y)
        return [cu_location_x,cu_location_y]
    def reset(self):
        # layer=random.randint(0,self.ad_counter)
        self.current_location_x=self.ad_state_x
        self.current_location_y=self.ad_state_y
        self.current_step=0
        return self.get_state()

    def area_density(self,x,y,w,h):   #计算密度
        env = dict()
        x_list = []
        y_list = []
        folder_path = "Datas/VR_frame_50"
        file_list = os.listdir(folder_path)
        # print(os.path.join(folder_path, file_list[0]))
        file_path=os.path.join(folder_path,file_list[self.density_layer])
        # print(file_path)
        with open(file_path, newline='') as file:
            eye_data_text = file.readlines()
            for line in eye_data_text:
                eye_list = line.split(',')
                frame, forward_x, forward_y, eye_x, eye_y = int(eye_list[1]), float(eye_list[3]), float(
                    eye_list[4]), float(eye_list[6]), float(eye_list[7])
                env[frame] = {'frame': frame, 'forward_x': forward_x, 'forward_y': forward_y, 'eye_x': eye_x,
                              'eye_y': eye_x}
                x_list.append(float(eye_list[6]))
                y_list.append(float(eye_list[7]))

        coordinates = np.column_stack((x_list, y_list))

        # 示例眼动轨迹数据
        eye_tracking_data = np.array(coordinates)

        rectangle_center = (x, y)
        rectangle_width = w
        rectangle_height = h
        left_bound = rectangle_center[0] - rectangle_width / 2
        right_bound = rectangle_center[0] + rectangle_width / 2
        top_bound = rectangle_center[1] + rectangle_height / 2
        bottom_bound = rectangle_center[1] - rectangle_height / 2

        # 划定区域的边界
        region_bounds = [(left_bound, bottom_bound), (right_bound, top_bound)]  # 替换为实际的区域边界

        # 筛选出划定区域内的眼动轨迹数据
        region_data = eye_tracking_data[(eye_tracking_data[:, 0] >= region_bounds[0][0]) &
                                        (eye_tracking_data[:, 0] <= region_bounds[0][1]) &
                                        (eye_tracking_data[:, 1] >= region_bounds[1][0]) &
                                        (eye_tracking_data[:, 1] <= region_bounds[1][1])]

        if region_data.size==0:
            return 0
        else:
            # 应用核密度估计
            kde = KernelDensity(bandwidth=0.1)
            kde.fit(region_data)

            # 计算密度值
            density_values = np.exp(kde.score_samples(region_data))

            # 计算密度大小
            total_density = np.sum(density_values)
            average_density = np.mean(density_values)

            # 生成密度图
            x, y = np.meshgrid(np.linspace(region_bounds[0][0], region_bounds[0][1], 100),
                               np.linspace(region_bounds[1][0], region_bounds[1][1], 100))
            grid_points = np.column_stack([x.ravel(), y.ravel()])
            density_values = np.exp(kde.score_samples(grid_points))
            density_map = density_values.reshape(100, 100)

            # 可视化密度图和划定区域
            # plt.contourf(x, y, density_map, cmap='viridis', levels=20)
            # plt.scatter(region_data[:, 0], region_data[:, 1], c='red', s=10, edgecolor='black')
            # plt.title('Eye Tracking Density in Defined Region')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Y Coordinate')
            # plt.show()

            # 输出密度大小
            # print(f"Total Density in the Defined Region: {total_density}")
            # print(f"Average Density in the Defined Region: {average_density}")

            return total_density


