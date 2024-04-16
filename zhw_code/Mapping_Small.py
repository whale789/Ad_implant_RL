
from math import sqrt, atan2, degrees
import os

from matplotlib import pyplot as plt
import kernel_density_estimation as dw

def plot_bar_chart(categories, values1, values2, title, xlabel, ylabel):
    x = range(len(categories))

    plt.bar(x, values1, width=0.4, label='植入前', align='center',color='cyan')
    plt.bar([i + 0.4 for i in x], values2, width=0.4, label='植入后', align='center',color='magenta')

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

def count_sum(ad_id,x,y,w,h):
    count=0
    for index in data_fin:
        if ad_id==index[0]:
            # print(index[0])
            if x-w/2<=index[1]<=x+w/2 and y-h/2<=index[2]<=y+h/2:
                count+=1
    return count


folder_path = "Experiment_Data_Test"
file_list = os.listdir(folder_path)
file_path = os.path.join(folder_path, file_list[0])
x_list = []
y_list=[]
count1 = 0
count2 = 0
data_fin=[]
count1_list=[]
count2_list=[]
with open(file_path, newline='', encoding='gb18030', errors='ignore') as file:
    eye_data_text = file.readlines()
    env = dict()
    i = 0
    for line in eye_data_text:
        # 解析数据中的旋转欧拉角（弧度）
        _, ad_id, _, _, _, rotation_x, rotation_y, rotation_z, _, left_eye_x, left_eye_y, _, right_eye_x, right_eye_y = line.split(",")

        # 给定的四元数x, y, z值
        rotation_x, rotation_y, rotation_z = float(rotation_x), float(rotation_y), float(rotation_z)
        # 计算w分量
        rotation_w = sqrt(1 - rotation_x**2 - rotation_y**2 - rotation_z**2)
        # 使用完整的四元数重新计算欧拉角
        yaw = atan2(2.0 * (rotation_w * rotation_y + rotation_z * rotation_x),
                    1 - 2 * (rotation_x**2 + rotation_y**2))
        pitch = atan2(2.0 * (rotation_w * rotation_x + rotation_y * rotation_z),
                      1 - 2 * (rotation_x**2 + rotation_z**2))
        roll = atan2(2.0 * (rotation_w * rotation_z + rotation_x * rotation_y),
                     1 - 2 * (rotation_y**2 + rotation_z**2))

        # 转换为度
        yaw_deg = degrees(yaw)
        pitch_deg = degrees(pitch)
        roll_deg = degrees(roll)
        # 重新使用修正的角度来计算ERP图像上的像素坐标
        x_pixel = ((yaw_deg + 180) / 360) * 8192
        y_pixel = ((pitch_deg + 90) / 180) * 4096
        # if ad_id=="ad7":
        #     print("坐标映射为：",x_pixel/8192, y_pixel/4096)
        # 处理每个眼动数据点
        erp_coordinates = []
        window_width=1920
        window_height=1080
        eye_data_samples=[(float(left_eye_x), float(left_eye_y)),(float(right_eye_x), float(right_eye_y))]
        for eye_x, eye_y in eye_data_samples:
            # 映射到1920x1080窗口中
            eye_window_x = ((eye_x + 1) / 2) * window_width
            eye_window_y = ((eye_y + 1) / 2) * window_height

            # 将局部窗口中点相对于FOV中心的偏移比例转换为ERP图中的像素偏移
            offset_ratio_x = (eye_window_x - window_width / 2) / window_width
            offset_ratio_y = (eye_window_y - window_height / 2) / window_height
            erp_offset_x = offset_ratio_x * 2276
            erp_offset_y = offset_ratio_y * 2276

            # 计算ERP图中的最终坐标
            final_erp_x = x_pixel + erp_offset_x
            final_erp_y = y_pixel + erp_offset_y  # ERP图的y坐标与像素坐标相反
            erp_coordinates.append((final_erp_x, final_erp_y))
            # 0.43, 0.5, 0.04, 0.053 ad4 √
            # 0.12, 0.55, 0.03, 0.08 ad5 x
            # 0.55, 0.5, 0.07, 0.21 ad6 √
            # 0.282, 0.4, 0.0255, 0.0714 ad7 x
            # 0.44, 0.45, 0.04, 0.15 ad2 √
            # 0.425, 0.5, 0.01, 0.05 ad3 √
            # 0.45, 0.455, 0.02, 0.08 √

        x_fin=(erp_coordinates[0][0]+erp_coordinates[1][0])/2/8192
        y_fin=(erp_coordinates[0][1]+erp_coordinates[1][1])/2/4096
        data_fin.append((ad_id,x_fin,y_fin))
        if ad_id=="ad1" or ad_id=="ad81":
            x_list.append(x_fin)
            y_list.append(y_fin)
dw.draw_point(x_list,y_list)
ad_list = ["ad1", "ad2", "ad3", "ad4",'ad6']
#
ad_xy_list=[[0.438, 0.465, 0.015, 0.115],[0.461, 0.465, 0.04, 0.15],[0.427, 0.41, 0.015, 0.05],
            [0.43, 1-0.547, 0.04, 0.125],[0.565, 0.45, 0.05, 0.15]]
add_ad_list = ["ad81", "ad82", "ad83", "ad84", 'ad86',]
# ad_list=["ad1","ad2","ad3","ad4",'ad5','ad6','ad7']
# ad_xy_list=[[0.45,0.455,0.02,0.08],[0.44,0.45,0.04,0.15],[0.425,0.53,0.01,0.05],[0.43,0.5,0.04,0.053],[0.12,0.54,0.03,0.08],[0.55,0.5,0.07,0.21],[0.282,0.4,0.0255,0.0714]]
# add_ad_list=["ad81","ad82","ad83","ad84",'ad85','ad86','ad87']
for i in range(len(ad_list)):
    count1=count_sum(ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
    count2=count_sum(add_ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
    print(ad_list[i],count1)
    print(add_ad_list[i],count2)
    count1_list.append(count1)
    count2_list.append(count2)

plot_bar_chart(ad_list, count1_list, count2_list, "眼动点数", "广告序号", "眼动点个数")
# print(data_fin)
