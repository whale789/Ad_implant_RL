import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

folder_path = "Datas/Experiment_Data"
file_list = os.listdir(folder_path)
file_path = os.path.join(folder_path, file_list[12])
x_list = []
with open(file_path, newline='', encoding='gb18030', errors='ignore') as file:
    eye_data_text = file.readlines()
    env = dict()

    i = 0
    for line in eye_data_text:
        eye_list = line.split(',')
        ad = eye_list[1]
        left_eye_x, left_eye_y, right_eye_x, right_eye_y = float(eye_list[8]), float(
            eye_list[9]), float(eye_list[11]), float(eye_list[11])
        env[i] = {"ad": ad, 'left_eye_x': left_eye_x, 'left_eye_y': left_eye_y, 'right_eye_x': right_eye_x,
                  'right_eye_y': right_eye_y}
        if ad == "ad85":
            # print("ad")
            # x_list.append(float(eye_list[8]))
            # y_list.append(float(eye_list[9]))
            # x_list.append(float(eye_list[11]))
            # y_list.append(float(eye_list[12]))
            x_list.append([(float(eye_list[8])+float(eye_list[11]))/2, (float(eye_list[9])+float(eye_list[12]))/2])
        i += 1
# 假设这是你的眼动数据，格式为 (x, y) 的列表，已经在 -1 到 1 的范围内
eye_data = x_list

# 创建坐标轴
fig, ax = plt.subplots()
ax.set_title("Eye Tracking Data")
ax.set_xlabel("X")
ax.set_ylabel("Y")


# 绘制眼动数据点
for x, y in eye_data:
    ax.scatter(x, y,s=15, label="Eye Movement", color='r')

# 绘制坐标轴
# ax.axhline(0, color='black', linewidth=0.5)
# ax.axvline(0, color='black', linewidth=0.5)

# 显示图例
# ax.legend()
rect = Rectangle((0.07,0.375), 0.02*2, 0.08*2, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(rect)

# 显示图形
plt.show()
