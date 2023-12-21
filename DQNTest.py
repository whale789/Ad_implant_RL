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
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

env=dict()
layer_count = 1
interval = 10
total_step = 1000
layer=1
x_list=[]
y_list=[]
xx_list=[]
yy_list=[]
with open("Datas/VR_frame/RL_023_frame/RL_frame25.txt",newline='') as file:
    eye_data_text=file.readlines()
    for line in eye_data_text:
        eye_list=line.split(',')
        frame,forward_x,forward_y,eye_x,eye_y=int(eye_list[1]),float(eye_list[3]),float(eye_list[4]),float(eye_list[6]),float(eye_list[7])
        env[frame]={'frame':frame,'forward_x':forward_x,'forward_y':forward_y,'eye_x':eye_x,'eye_y':eye_x}
        x_list.append(float(eye_list[6]))
        y_list.append(float(eye_list[7]))

coordinates = np.column_stack((x_list, y_list))
print(coordinates)


# 示例眼动轨迹数据
eye_tracking_data = np.array(coordinates)

# 离散化数据
grid_x, grid_y = np.mgrid[min(eye_tracking_data[:, 0]):max(eye_tracking_data[:, 0]):100j,
                          min(eye_tracking_data[:, 1]):max(eye_tracking_data[:, 1]):100j]

# 将轨迹点映射到网格上
points = np.vstack([grid_x.ravel(), grid_y.ravel()])
kde = KernelDensity(bandwidth=0.3)
kde.fit(eye_tracking_data)
density = np.exp(kde.score_samples(points.T))

# 将密度结果可视化
plt.pcolormesh(grid_x, grid_y, density.reshape(grid_x.shape), cmap='viridis')
plt.scatter(eye_tracking_data[:, 0], eye_tracking_data[:, 1], c='red', s=50, edgecolor='black')
plt.title('Eye Tracking Density Estimation')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.colorbar(label='Density')
plt.show()