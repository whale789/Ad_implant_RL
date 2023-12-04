import copy
import random

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class Ad_Environment:
    def __init__(self,ad_state_x,ad_state_y,layer,ad_counter,ad_width,ad_height,total_step,ad_density):
        self.layer=layer  #广告状态空间索引
        self.ad_counter=ad_counter-1   #广告空间总数
        self.ad_state_x=ad_state_x
        self.ad_state_y=ad_state_y
        self.ad_location_x = ad_state_x[int(self.layer)]  # 广告水平位置
        # print("00x",self.ad_location_x)
        self.ad_location_y = ad_state_y[int(self.layer)]  # 广告垂直位置
        # print("00y",self.ad_location_y)
        self.ad_width=ad_width   #所植入广告的宽度
        self.ad_height=ad_height  #所植入广告的高度
        self.total_step=total_step   ##总步数
        self.current_step=0  #当前步数
        self.ad_density=ad_density  #初始位置密度
        self.action_space=[0,1,2,3]  #分别代表up,down,left,right
        self.current_location_x=self.ad_location_x
        self.current_location_y=self.ad_location_y

    def step(self,action):
        if not action in self.action_space:
            print("该Action不存在")
        else:
            # print("111:",self.ad_state_x)
            # print("444",self.layer,self.ad_counter)
            self.layer+=1
            if self.layer>self.ad_counter:
                self.layer=self.layer%self.ad_counter
            self.current_location_x = self.ad_state_x[self.layer]
            self.current_location_y = self.ad_state_y[self.layer]
            # print("222",self.current_location_x)

        # if action==0:  #平移
        #     # self.current_location_y+=1
        #     self.current_location_x=self.ad_location_x[(self.layer+1)%self.ad_counter]
        #     self.current_location_y=self.ad_location_y[(self.layer+1)%self.ad_counter]
        # elif action==1:  #缩放
        #     # self.current_location_y-=1
        #     pass
        # elif action==2:   #旋转
        #     # self.current_location_x-=1
        #     pass



        reward=self.calculate_reward()   #奖励函数

        self.current_step+=1
        done=self.current_step>=self.total_step

        return (self.current_location_x,self.current_location_y),reward,done

    def calculate_reward(self):
        ideal_location_x=np.mean(self.ad_location_x)   #计算平均值？
        iddal_location_y=np.mean(self.ad_location_y)

        #2023.12.1方案：根据广告植入区域的密度进行计算
        #状态为广告的中心点，以及广告的宽度及高度，以此进而对缩放也加以实现
        #reward设为对该区域的密度
        reward=0
        density=self.area_density(self.current_location_x,self.current_location_y,self.ad_width,self.ad_height)  #计算该区域的密度
        density_difference=density-self.ad_density
        reward+= round(density_difference, 4)
        print('333',reward)
        self.ad_density=density

        return reward

    def get_state(self):   #讲ad_location_x和ad_location_y拼接起来
        cu_location_x=copy.copy(self.current_location_x)
        cu_location_y=copy.copy(self.current_location_y)
        return [cu_location_x,cu_location_y]
    def reset(self):
        layer=random.randint(0,self.ad_counter)
        self.current_location_x=self.ad_state_x[layer]
        self.current_location_y=self.ad_state_y[layer]
        self.current_step=0
        return self.get_state()

    def area_density(self,x,y,w,h):   #计算密度
        env = dict()
        x_list = []
        y_list = []
        with open("Datas/Gaze_txt_files/p001/179._2017-10-13-10-27_ori_0.txt", newline='') as file:
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
            print(f"Total Density in the Defined Region: {total_density}")
            # print(f"Average Density in the Defined Region: {average_density}")

            return total_density


