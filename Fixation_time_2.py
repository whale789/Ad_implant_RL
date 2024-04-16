
from datetime import datetime
from math import sqrt, atan2, degrees
import os
from matplotlib import pyplot as plt

def plot_bar_chart(categories, values1, values2, title, xlabel, ylabel):
    x = range(len(categories))

    plt.bar(x, values1, width=0.4, label='植入前', align='center')
    plt.bar([i + 0.4 for i in x], values2, width=0.4, label='植入后', align='center')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([i for i in x], categories)
    plt.legend()

    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.show()
def time_fixate(time1,time2):
    times= [time1, time2]
    # times=["4月1日12:19:40:983","4月1日12:19:42:221"]
    time_format = "%m月%d日%H:%M:%S:%f"
    datetime_objects = [datetime.strptime(time, time_format) for time in times]
    time_diffs_ms = (datetime_objects[1] - datetime_objects[0]).total_seconds()
    # print(time_diffs_ms)
    return time_diffs_ms


def count_time(ad_id,x,y,w,h):
    time_c=0
    for ii in range(len(data_fin)-1):
        if data_fin[ii][0]==ad_id:
            if x-w/2<=data_fin[ii][1]<=x+w/2 and y-h/2<=data_fin[ii][2]<=y+h/2:
                time_c+=time_fixate(data_fin[ii][3],data_fin[ii+1][3])
    return time_c
def first_time(ad_id,x,y,w,h):
    f_time=""
    for j in range(len(data_fin) - 1):
        if data_fin[j][0]==ad_id:
            f_time = data_fin[j][3]
            break
    for ii in range(len(data_fin)-1):
        if data_fin[ii][0]==ad_id:
            if x-w/2<=data_fin[ii][1]<=x+w/2 and y-h/2<=data_fin[ii][2]<=y+h/2:
                return time_fixate(f_time,data_fin[ii][3])
def number_of_visits(ad_id,x,y,w,h):
    s=0
    ss=0
    for ii in range(len(data_fin)-1):
        if data_fin[ii][0]==ad_id:
            ss+=1
            if data_fin[ii][1]<x-w/2 or data_fin[ii][1]>x+w/2 or data_fin[ii][2]>y+h/2 or data_fin[ii][2]<y-h/2:
                if x-w/2<=data_fin[ii+1][1]<=x+w/2 and y-h/2<=data_fin[ii+1][2]<=y+h/2:
                    s=s+1
    return s/ss

# print(objects)
data_fin=[]
folder_path = "Datas/Experiment_Data"
ad_time_count = [0,0,0,0,0]
add_ad_time_count=[0,0,0,0,0]
f_ad_time=[0,0,0,0,0]
s_ad_time=[0,0,0,0,0]
time_count1_list = []
time_count2_list = []
first_ad_time = []
file_count=0
ad_list = ["ad1", "ad2", "ad3", "ad4", 'ad6']
# ad_xy_list = [[0.45, 0.455, 0.02, 0.08], [0.44, 0.45, 0.04, 0.15], [0.425, 0.53, 0.01, 0.05],
#               [0.43, 0.5, 0.04, 0.053], [0.55, 0.5, 0.07, 0.21]]
ad_xy_list = [[0.445, 0.465, 0.03, 0.23], [0.442, 0.465, 0.08, 0.3], [0.435, 0.55, 0.03, 0.1],
                 [0.43, 0.53, 0.035, 0.055], [0.565, 0.52, 0.1, 0.3]]

add_ad_list = ["ad81", "ad82", "ad83", "ad84", 'ad86', ]
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    file_count += 1
    with open(file_path, newline='', encoding='utf-8', errors='ignore') as file:
        eye_data_text = file.readlines()
        env = dict()
        i = 0
        for line in eye_data_text:
            # 解析数据中的旋转欧拉角（弧度）
            ad_time, ad_id, _, _, _, rotation_x, rotation_y, rotation_z, _, left_eye_x, left_eye_y, _, right_eye_x, right_eye_y = line.split(",")

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
            data_fin.append((ad_id,x_fin,y_fin,ad_time))

    for i in range(len(ad_list)):
        count1=count_time(ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
        count2=count_time(add_ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
        ad_time_count[i]=+count1
        add_ad_time_count[i]=+count2
        f_ad_time[i]+=first_time(add_ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
        s_ad_time[i]+=number_of_visits(add_ad_list[i],ad_xy_list[i][0],ad_xy_list[i][1],ad_xy_list[i][2],ad_xy_list[i][3])
        time_count1_list.append(count1)
        time_count2_list.append(count2)
for i in range(len(ad_time_count)):
    ad_time_count[i]=ad_time_count[i]/file_count
    add_ad_time_count[i]=add_ad_time_count[i]/file_count
    f_ad_time[i]=f_ad_time[i]/file_count
    s_ad_time[i]=s_ad_time[i]
print(add_ad_time_count)
print(f_ad_time)
print(s_ad_time)
plot_bar_chart(ad_list, ad_time_count, add_ad_time_count, "注视时间对比", "广告序号", "注视时间  /秒")