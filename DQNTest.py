# import numpy as np
# from sklearn.neighbors import KernelDensity
# import matplotlib.pyplot as plt
#
#
# env=dict()
# layer_count = 1
# interval = 10
# total_step = 1000
# layer=1
# x_list=[]
# y_list=[]
# xx_list=[]
# yy_list=[]
# with open("Datas/VR_frame/RL_023_frame/RL_frame25.txt",newline='') as file:
#     eye_data_text=file.readlines()
#     for line in eye_data_text:
#         eye_list=line.split(',')
#         frame,forward_x,forward_y,eye_x,eye_y=int(eye_list[1]),float(eye_list[3]),float(eye_list[4]),float(eye_list[6]),float(eye_list[7])
#         env[frame]={'frame':frame,'forward_x':forward_x,'forward_y':forward_y,'eye_x':eye_x,'eye_y':eye_x}
#         x_list.append(float(eye_list[6]))
#         y_list.append(float(eye_list[7]))
#
# coordinates = np.column_stack((x_list, y_list))
#
# # 示例眼动轨迹数据
# eye_tracking_data = np.array(coordinates)
#
# # 核密度估计
# kde = KernelDensity(bandwidth=0.2)
# kde.fit(eye_tracking_data)
#
# # 生成密度图
# x, y = np.meshgrid(np.linspace(min(eye_tracking_data[:, 0]), max(eye_tracking_data[:, 0]), 100),
#                    np.linspace(min(eye_tracking_data[:, 1]), max(eye_tracking_data[:, 1]), 100))
# grid_points = np.column_stack([x.ravel(), y.ravel()])
# density_values = np.exp(kde.score_samples(grid_points))
#
# # 设置密度阈值
# threshold = 6
#
# # 根据阈值找到密度最大区域
# max_density_region = grid_points[density_values > threshold]
#
# center_point = np.mean(max_density_region, axis=0)
#
# # 绘制密度图和密度最大区域
# plt.contourf(x, y, density_values.reshape(100, 100), cmap='viridis', levels=20)
# plt.scatter(eye_tracking_data[:, 0], eye_tracking_data[:, 1], c='red', s=50, edgecolor='black')
# plt.scatter(max_density_region[:, 0], max_density_region[:, 1], c='blue', s=50, edgecolor='white', label='Max Density Region')
# plt.scatter(center_point[0],center_point[1],c='green',edgecolors='black')
# plt.title('Eye Tracking Density Estimation')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
# plt.show()
#
# # 输出密度最大区域的数据
# print("中心点为:")
# print(center_point)





# import numpy as np
# from sklearn.neighbors import KernelDensity
# import matplotlib.pyplot as plt
#
# env=dict()
# layer_count = 1
# interval = 10
# total_step = 1000
# layer=1
# x_list=[]
# y_list=[]
# xx_list=[]
# yy_list=[]
# with open("Datas/VR_frame_50/RL_025_50frame.txt",newline='') as file:
#     eye_data_text=file.readlines()
#     for line in eye_data_text:
#         eye_list=line.split(',')
#         frame,forward_x,forward_y,eye_x,eye_y=int(eye_list[1]),float(eye_list[3]),float(eye_list[4]),float(eye_list[6]),float(eye_list[7])
#         env[frame]={'frame':frame,'forward_x':forward_x,'forward_y':forward_y,'eye_x':eye_x,'eye_y':eye_x}
#         x_list.append(float(eye_list[6]))
#         y_list.append(float(eye_list[7]))
#
# coordinates = np.column_stack((x_list, y_list))
# print(coordinates)
#
#
# # 示例眼动轨迹数据
# eye_tracking_data = np.array(coordinates)
#
# # 离散化数据
# grid_x, grid_y = np.mgrid[min(eye_tracking_data[:, 0]):max(eye_tracking_data[:, 0]):100j,
#                           min(eye_tracking_data[:, 1]):max(eye_tracking_data[:, 1]):100j]
#
# # 将轨迹点映射到网格上
# points = np.vstack([grid_x.ravel(), grid_y.ravel()])
# kde = KernelDensity(bandwidth=0.3)
# kde.fit(eye_tracking_data)
# density = np.exp(kde.score_samples(points.T))
#
# # 将密度结果可视化
# plt.pcolormesh(grid_x, grid_y, density.reshape(grid_x.shape), cmap='viridis')
# plt.scatter(eye_tracking_data[:, 0], eye_tracking_data[:, 1], c='red', s=50, edgecolor='black')
# plt.title('Eye Tracking Density Estimation')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.colorbar(label='Density')
# plt.show()







import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from decimal import Decimal, getcontext

# 设置精度和舍入模式
getcontext().prec = 10  # 设置精度为10位
getcontext().rounding = "ROUND_HALF_UP"  # 设置舍入模式为四舍五入
env = dict()
x_list = []
y_list = []
folder_path = "Datas/VR_frame_50"
file_list = os.listdir(folder_path)
# print(os.path.join(folder_path, file_list[0]))
file_path = os.path.join(folder_path, file_list[4])
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
# eye_tracking_data = np.random.rand(2, 100)
# 示例眼动轨迹数据
eye_tracking_data = np.array(coordinates)
# print(eye_tracking_data)
x=0.45
y=0.5
w=0.2
h=0.2
rectangle_center = (x, y)
rectangle_width = w
rectangle_height = h
left_bound = rectangle_center[0] - rectangle_width / 2
right_bound = rectangle_center[0] + rectangle_width / 2
top_bound = rectangle_center[1] + rectangle_height / 2
bottom_bound = rectangle_center[1] - rectangle_height / 2

# 划定区域的边界
region_bounds = [(left_bound, bottom_bound), (right_bound, top_bound)]  # 替换为实际的区域边界
print(region_bounds)
print(left_bound)
# 筛选出划定区域内的眼动轨迹数据
# print(eye_tracking_data[:,0])
# print(eye_tracking_data[:,1])
# print(region_bounds[1][0])
region_data = eye_tracking_data[(eye_tracking_data[:, 0] >= region_bounds[0][0]) &
                                (eye_tracking_data[:, 0] <= region_bounds[1][0]) &
                                (eye_tracking_data[:, 1] >= region_bounds[0][1]) &
                                (eye_tracking_data[:, 1] <= region_bounds[1][1])]
# region_data = eye_tracking_data[eye_tracking_data[:, 0] >= region_bounds[0][0]]
# for point in eye_tracking_data:
#     # print(point[0])
#     if Decimal(left_bound)<=Decimal(point[0]):
#         print(point[0])


# print(region_data)
if region_data.size == 0:
    print("密度为0")
else:
    # 应用核密度估计
    kde = KernelDensity(bandwidth=0.3)
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
    plt.contourf(x, y, density_map, cmap='viridis', levels=20)
    plt.scatter(region_data[:, 0], region_data[:, 1], c='red', s=10, edgecolor='black')
    plt.title('Eye Tracking Density in Defined Region')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    # 输出密度大小
    print(f"Total Density in the Defined Region: {total_density}")
    print(f"Average Density in the Defined Region: {average_density}")


