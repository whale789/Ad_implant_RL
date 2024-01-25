import copy
import os
import random

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
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
        self.aspect_ratio=ad_height/ad_width
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
            # self.current_location_x = self.ad_state_x
            # self.current_location_y = self.ad_state_y
            # print("222",self.current_location_x)
            if action==0:  #向上平移
               self.current_location_x=self.current_location_x
               self.current_location_y=self.current_location_y+0.001
            elif action==1:  #向下平移
                self.current_location_x=self.current_location_x
                self.current_location_y=self.current_location_y-0.001
            elif action==2:  #向左平移
                self.current_location_x=self.current_location_x-0.001
                self.current_location_y=self.current_location_y
            elif action==3:     #向右平移
                self.current_location_x=self.current_location_x+0.001
                self.current_location_y=self.current_location_y
            elif action==4:  #放大
                self.current_width=self.current_width+0.0005
                self.current_height=self.current_width*self.aspect_ratio
            elif action==5:  #缩小
                self.current_width=self.current_width-0.0005
                self.current_height=self.current_width*self.aspect_ratio
            # elif action==6:   #旋转
            #     # self.current_location_x-=1
            #     pass
            # print("111",self.current_location_x,self.current_location_y)
            # print("222",self.current_width,self.current_height)



        reward=self.calculate_reward()   #奖励函数

        self.current_step+=1
        # done=self.current_step>=self.total_step
        done=False
        if self.current_step>=self.total_step:
            done=True
            # print(self.density_layer)
            # self.density_layer=(self.density_layer+1)%20
            self.density_layer=self.density_layer+1
        return (self.current_location_x,self.current_location_y,self.current_width,self.current_height),reward,done

    def calculate_reward(self):
        #根据中心点和宽度、高度计算是否超出了限制区域
        if self.current_location_x+(self.current_width/2)>self.ad_limit_x+(self.ad_limit_width/2):
            self.total_reward=((self.current_location_x+self.current_width/2)-(self.ad_limit_x+self.ad_limit_width/2))*(-10)
            # print("111",self.total_reward)
        elif self.current_location_y+self.current_height/2>self.ad_limit_y+self.ad_limit_height/2:
            self.total_reward=((self.current_location_y+self.current_height/2)-(self.ad_limit_y+self.ad_limit_height/2))*(-10)
            # print("222", self.total_reward)
        elif self.current_location_x-self.current_width/2<self.ad_limit_x-self.ad_limit_width/2:
            self.total_reward=((self.current_location_x-self.current_width/2)-(self.ad_limit_x-self.ad_limit_width/2))*10
            # print("333", self.total_reward)
        elif self.current_location_y-self.current_height/2<self.ad_limit_y-self.ad_limit_height/2:
            self.total_reward=((self.current_location_y-self.current_height/2)-(self.ad_limit_y-self.ad_limit_height/2))*10
            # print("444", self.total_reward)
        elif self.salience_area(self.current_location_x, self.current_location_y, self.current_width,
                                        self.current_height)==0:
            self.total_reward=-100
        else:
            density = self.area_density_2(self.current_location_x, self.current_location_y, self.current_width,
                                        self.current_height)  # 计算该区域的密度
            # print("密度是：",density)
            density_difference = density - self.ad_density
            # if density_difference>0:
            #     density_difference = density_difference*5
            # self.total_reward = round(self.total_reward + round(density_difference, 12) / 100, 12)
            self.total_reward=density_difference
            self.ad_density = density
            # print("555",self.total_reward)
            # print("密度差异为：",density_difference)
        # print(self.total_reward)
        return self.total_reward

    def get_state(self):   #讲ad_location_x和ad_location_y拼接起来
        cu_location_x=copy.copy(self.current_location_x)
        cu_location_y=copy.copy(self.current_location_y)
        cu_location_width=copy.copy(self.current_width)
        cu_location_height=copy.copy(self.current_height)
        return [cu_location_x,cu_location_y,cu_location_width,cu_location_height]
    def reset(self):
        # layer=random.randint(0,self.ad_counter)
        self.current_location_x=self.ad_state_x
        self.current_location_y=self.ad_state_y
        self.current_width=self.ad_width
        self.current_height=self.ad_height
        self.current_step=0
        return self.get_state()

    def area_density(self,x,y,w,h):   #计算密度
        env = dict()
        x_list = []
        y_list = []
        folder_path = "Datas/VR_frame_50"
        file_list = os.listdir(folder_path)
        # print(os.path.join(folder_path, file_list[0]))
        # file_path=os.path.join(folder_path,file_list[self.density_layer])   #train
        file_path = os.path.join(folder_path, file_list[20])  # test
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

            return round(average_density,12)

    def area_density_2(self, x, y, w, h):  # 计算密度
        if w==0 or h==0:
            return 0
        else:
            env = dict()
            x_list = []
            y_list = []
            folder_path = "Datas/Gaze_files"
            file_list = os.listdir(folder_path)
            # print(os.path.join(folder_path, file_list[0]))
            # file_path=os.path.join(folder_path,file_list[self.density_layer])   #train
            file_path = os.path.join(folder_path, file_list[21])  # test

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

            if region_data.size == 0:
                return 0
            else:
                # print(region_data.size / (w * 100 * h * 100))
                return region_data.size / (w * 100 * h * 100)

    def salience_area(self,x,y,w,h):   #显著性区域
        if x==0 or y==0 or w==0 or h==0:
            return 1
        else:
            env = dict()
            x_list = []
            y_list = []
            folder_path = "Datas/VR_frame_50"
            file_list = os.listdir(folder_path)
            # print(os.path.join(folder_path, file_list[0]))
            # file_path = os.path.join(folder_path, file_list[self.density_layer])
            file_path = os.path.join(folder_path, file_list[21])  # test
            with open(file_path, newline='') as file:
                eye_data_text = file.readlines()
                for line in eye_data_text:
                    eye_list = line.split(',')
                    frame, forward_x, forward_y, eye_x, eye_y = int(eye_list[1]), float(eye_list[3]), float(
                        eye_list[4]), float(eye_list[6]), float(eye_list[7])
                    env[frame] = {'frame': frame, 'forward_x': forward_x, 'forward_y': forward_y, 'eye_x': eye_x,
                                  'eye_y': eye_x}
                    x_list.append([float(eye_list[6]), float(eye_list[7])])
            # 模拟眼动轨迹数据（归一化坐标）
            eye_tracking_data = np.array(x_list)

            # 图像大小
            image_size = (500, 1000)  # 假设全景图大小为 (height, width)

            # 生成眼动热度图
            heat_map = np.zeros(image_size)
            for gaze_point in eye_tracking_data:
                x, y = int(gaze_point[1] * image_size[1]), int((1 - gaze_point[0]) * image_size[0])
                heat_map[y, x] += 1

            # 空间平滑
            heat_map_smoothed = gaussian_filter(heat_map, sigma=5)

            # 归一化
            normalized_heat_map = (heat_map_smoothed - heat_map_smoothed.min()) / (
                        heat_map_smoothed.max() - heat_map_smoothed.min())

            # 自适应阈值设置
            adaptive_threshold = np.mean(normalized_heat_map) + np.std(normalized_heat_map)
            # 定义一个矩形区域 [y_min, y_max, x_min, x_max]，已归一化
            rectangle_bbox_normalized = [y - h / 2, y + h / 2, x - w / 2, x + w / 2]

            # 转换矩形区域坐标为图像坐标
            rectangle_bbox_image = [
                int(rectangle_bbox_normalized[0] * image_size[0]),
                int(rectangle_bbox_normalized[1] * image_size[0]),
                int(rectangle_bbox_normalized[2] * image_size[1]),
                int(rectangle_bbox_normalized[3] * image_size[1])
            ]

            # 提取矩形区域内的显著性信息
            rectangle_heat_map = normalized_heat_map[rectangle_bbox_image[1]:rectangle_bbox_image[0],
                                 rectangle_bbox_image[2]:rectangle_bbox_image[3]]

            # 判断是否有重合区域
            overlap = np.any(rectangle_heat_map > adaptive_threshold)

            # 显示结果
            # cv2.imshow('Normalized Heat Map', (normalized_heat_map * 255).astype(np.uint8))
            # cv2.rectangle(normalized_heat_map, (rectangle_bbox_image[2], rectangle_bbox_image[1]),
            #               (rectangle_bbox_image[3], rectangle_bbox_image[0]), (255, 255, 255), 2)

            if overlap:
                # print("矩形区域与显著性区域有重合")
                return 0
            else:
                # print("矩形区域与显著性区域无重合")
                return 1
            # cv2.imshow('Rectangle Region', normalized_heat_map)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
