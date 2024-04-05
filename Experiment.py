import math
import os
from decimal import getcontext
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.pyplot as plt
getcontext().prec = 10  # 设置精度为10位
getcontext().rounding = "ROUND_HALF_UP"  # 设置舍入模式为四舍五入


def plot_bar_chart(categories, values1, values2, title, xlabel, ylabel):
    x = range(len(categories))

    plt.bar(x, values1, width=0.4, label='植入前', align='center')
    plt.bar([i + 0.4 for i in x], values2, width=0.4, label='植入后', align='center')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([i + 0.2 for i in x], categories)
    plt.legend()

    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.show()


folder_path = "Datas/Experiment_Data"
file_list = os.listdir(folder_path)
file_path = os.path.join(folder_path, file_list[12])
def density_calculation(ad_id,x,y,w,h):
    # print(x,y,w,h)
    env = dict()
    x_list = []
    y_list = []
    with open(file_path, newline='',encoding='gb18030', errors='ignore') as file:
        eye_data_text = file.readlines()
        i=0
        for line in eye_data_text:
            eye_list = line.split(',')
            ad=eye_list[1]
            left_eye_x, left_eye_y, right_eye_x, right_eye_y = float(eye_list[8]), float(
                eye_list[9]), float(eye_list[11]), float(eye_list[12])
            env[i] = {"ad":ad,'left_eye_x': left_eye_x, 'left_eye_y': left_eye_y, 'right_eye_x': right_eye_x,
                          'right_eye_y': right_eye_y}
            if ad==ad_id:
                # print("ad")
                # x_list.append(float(eye_list[8]))
                # y_list.append(float(eye_list[9]))
                # x_list.append(float(eye_list[11]))
                # y_list.append(float(eye_list[12]))
                # x_list.append([float(eye_list[8]),float(eye_list[9])])
                # x_list.append([float(eye_list[11]),float(eye_list[12])])
                x1=(float(eye_list[8]) + float(eye_list[11])) / 2
                y1=(float(eye_list[9]) + float(eye_list[12])) / 2
                # x1=(x1-0.5)*(110/180)*np.pi
                # y1=(y1-0.5)*(110/180)*np.pi
                x_list.append([x1,y1])
            i+=1
    # 模拟眼动轨迹数据（归一化坐标）
    # coordinates = np.column_stack((x_list, y_list))
    # eye_tracking_data = np.array(coordinates)
    # img_path = 'Datas/Testing_scenarios/ad2.png'
    # img = Image.open(img_path)
    # img_width, img_height = img.size
    # x_left=(x-w/2)*2*np.pi-np.pi
    # y_bootom=(1-y-h/2)*np.pi-0.5*np.pi
    # x_right=(x+w/2)*2*np.pi-np.pi
    # y_top=(1-y+h/2)*np.pi-0.5*np.pi
    # print("球形坐标：",(x,y))
    x=x/6
    y=y-0.05
    rectangle_center = (x, y)
    rectangle_width = w
    rectangle_height = h
    left_bound = rectangle_center[0] - rectangle_width / 2
    right_bound = rectangle_center[0] + rectangle_width / 2
    top_bound = rectangle_center[1] + rectangle_height / 2
    bottom_bound = rectangle_center[1] - rectangle_height / 2
    # if x_left>x_right:
    #     x_left,x_right=x_right,x_left
    # if y_bootom>y_top:
    #     y_bottom,y_top=y_top,y_bootom
    # left_bound = x_left
    # right_bound = x_right
    # top_bound = y_top
    # bottom_bound = y_bootom

    # 划定区域的边界
    region_bounds = [(left_bound, bottom_bound), (right_bound, top_bound)]  # 替换为实际的区域边界
    # 筛选出划定区域内的眼动轨迹数据
    region_data=[]
    count=0
    for eye in x_list:
        # print(eye[0])
        if eye[0]>=region_bounds[0][0] and eye[0]<=region_bounds[1][0] and eye[1]>=region_bounds[0][1] and eye[1]<=region_bounds[1][1]:
            region_data.append([eye[0],eye[1]])
            count+=1
    # print(region_data)
    if len(region_data) == 0:
        # print("0")
        return 0,0
    else:
        # print(region_data.size / (w * 100 * h * 100))
        # print(count / (w*h))
        return count,count/(w*h)


ad_list=["ad1","ad2","ad3","ad4",'ad5','ad6','ad7']
ad_xy_list=[[0.45,0.455,0.02,0.08],[0.44,0.45,0.04,0.15],[0.425,0.53,0.01,0.05],[0.43,0.5,0.04,0.053],[0.12,0.54,0.03,0.08],[0.55,0.5,0.07,0.21],[0.282,0.4,0.0255,0.0714]]
add_ad_list=["ad81","ad82","ad83","ad84",'ad85','ad86','ad87']
ad_count=[]
ad_density=[]
for i in range(len(ad_list)):
    count,density=density_calculation(ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
    ad_count.append(count)
    ad_density.append(density)
    # print(count,density)
print(ad_density)
add_ad_count=[]
add_ad_density=[]
for i in range(len(add_ad_list)):
    count,density=density_calculation(add_ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
    add_ad_count.append(count)
    add_ad_density.append(density)
print(add_ad_density)
plot_bar_chart(ad_list, ad_density, add_ad_density, "密度对比", "广告", "密度")




#
#
# # 示例数据
# categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
# values1 = [10, 15, 7, 10]
# values2 = [12, 10, 6, 15]
# title = 'Comparison of Two Bars'
# xlabel = 'Categories'
# ylabel = 'Values'

# plot_bar_chart(categories, values1, values2, title, xlabel, ylabel)




