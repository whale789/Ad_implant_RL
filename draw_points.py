import os
from math import sqrt, atan2, degrees

#全景图中画注视点
# from PIL import Image, ImageDraw
#
# # 加载全景图
# image_path = 'Datas/Testing_scenarios/ad1.png'  # 更换为你的全景图文件路径
# image = Image.open(image_path)
#
# # 创建一个可用于绘制的对象
# draw = ImageDraw.Draw(image)
# env=dict()
# x_list=[]
# # 归一化处理的眼动坐标点，范围从0到1
# eye_movement_points = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75)]
# with open("Datas/Gaze_files/100._2017-09-30-15-02_ori_0.txt",newline='') as file:
#     eye_data_text=file.readlines()
#     for line in eye_data_text:
#         eye_list=line.split(',')
#         frame,forward_x,forward_y,eye_x,eye_y=int(eye_list[1]),float(eye_list[3]),float(eye_list[4]),float(eye_list[6]),float(eye_list[7])
#         env[frame]={'frame':frame,'forward_x':forward_x,'forward_y':forward_y,'eye_x':eye_x,'eye_y':eye_x}
#         x_list.append([float(eye_list[6]),float(eye_list[7])])
# eye_movement_points=x_list
# # 图像尺寸
# image_width, image_height = image.size
#
# # 绘制点
# for x_norm, y_norm in eye_movement_points:
#     x_pixel = x_norm * image_width
#     y_pixel = y_norm * image_height
#     # 绘制一个小圆来代表点
#     radius = 20  # 点的大小
#     draw.ellipse((x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius), fill='red')
#     # 绘制连接椭圆顶点的线段
#
# # 保存或显示图像
# # image.show()  # 直接显示
# # 或者保存到文件
# image.save('Draw_Points.jpg')







#坐标轴画注视路径
# import matplotlib.pyplot as plt
#
# env=dict()
# x_list=[]
# y_list=[]
#
# with open("Datas/Gaze_files/100._2017-09-30-15-02_ori_0.txt",newline='') as file:
#     eye_data_text=file.readlines()
#     for line in eye_data_text:
#         eye_list=line.split(',')
#         frame,forward_x,forward_y,eye_x,eye_y=int(eye_list[1]),float(eye_list[3]),float(eye_list[4]),float(eye_list[6]),float(eye_list[7])
#         env[frame]={'frame':frame,'forward_x':forward_x,'forward_y':forward_y,'eye_x':eye_x,'eye_y':eye_x}
#         x_list.append(float(eye_list[6]))
#         y_list.append(float(eye_list[7]))
#
# # 假设你有一组眼动坐标数据
# # eye_movement_data = [(10, 20), (30, 40), (50, 60), (70, 80)]
#
# # # 提取 x 和 y 坐标
# # x = [point[0] for point in eye_movement_data]
# # y = [point[1] for point in eye_movement_data]
# x=x_list
# y=y_list
# # 创建一个新的图形
# # plt.figure()
# plt.figure(figsize=(8, 20))
# # 创建坐标轴
# plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴比例为相等
# plt.xlabel('X轴')  # 设置X轴标签
# plt.ylabel('Y轴')  # 设置Y轴标签
#
# # 画出眼动坐标
# plt.plot(x, y, marker='o', linestyle='-', color='b')  # 使用圆点标记绘制眼动路径
#
# # 显示图形
# plt.show()





#全景图中画眼动路径
from PIL import Image, ImageDraw

# 加载全景图
# image_path = 'Datas/Testing_scenarios/ad6.png'  # 更换为你的全景图文件路径
image_path = 'Datas/Experiment_images/tyh/tyh_ad_6.png'
image = Image.open(image_path)

# 创建一个可用于绘制的对象
draw = ImageDraw.Draw(image)

# 归一化处理的眼动坐标点，范围从0到1
# eye_movement_points = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75)]
x_list=[]
folder_path = "zhw_code/Experiment_Data_Test"
file_list = os.listdir(folder_path)
file_path = os.path.join(folder_path, file_list[10])
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
        if ad_id=="ad86":
            x_list.append([x_fin,y_fin])
eye_movement_points=x_list
# 图像尺寸
image_width, image_height = image.size

# 绘制点和线
for i in range(len(eye_movement_points)):
    x_norm, y_norm = eye_movement_points[i]
    x_pixel = x_norm * image_width
    y_pixel = y_norm * image_height

    # 绘制点
    radius = 15  # 点的大小，减小以适应线的绘制
    draw.ellipse((x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius), fill='red')

    # 如果不是第一个点，绘制一条从前一个点到当前点的线
    if i > 0:
        prev_x_norm, prev_y_norm = eye_movement_points[i - 1]
        prev_x_pixel = prev_x_norm * image_width
        prev_y_pixel = prev_y_norm * image_height
        # 绘制线
        draw.line((prev_x_pixel, prev_y_pixel, x_pixel, y_pixel), fill='red', width=8)

# 保存或显示图像
# image.show()  # 直接显示
# 或者保存到文件
image.save('Draw_Points_3.jpg')
