#
#
# #显著性区域算法图片显示
# import os
# import cv2
# import time
# import glob
# from threading import Thread
# import numpy as np
#
#
# def read_image(image_path):
#     image = cv2.imread(image_path)
#     min_edge = min(image.shape[0], image.shape[1])  # 图片窄边
#     proportion = 1  # 缩放比例
#     if min_edge > 3000:
#         proportion = 0.1
#     elif 2000 < min_edge <= 3000:
#         proportion = 0.2
#     elif 1000 < min_edge <= 2000:
#         proportion = 0.3
#     elif 700 <= min_edge <= 1000:
#         proportion = 0.4
#     resize_image = cv2.resize(image, None, fx=proportion, fy=proportion, interpolation=cv2.INTER_CUBIC)
#     image_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
#     return image_gray
#
#
# def LC(image_gray):
#     image_height = image_gray.shape[0]
#     image_width = image_gray.shape[1]
#     image_gray_copy = np.zeros((image_height, image_width))
#     hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])  # 直方图，统计图像中每个灰度值的数量
#     gray_dist = cal_dist(hist_array)  # 灰度值与其他值的距离
#     print(gray_dist)
#     for i in range(image_width):
#         for j in range(image_height):
#             temp = image_gray[j][i]
#             image_gray_copy[j][i] = gray_dist[temp]
#     image_gray_copy = (image_gray_copy - np.min(image_gray_copy)) / (np.max(image_gray_copy) - np.min(image_gray_copy))
#     return image_gray_copy
#
#
# def cal_dist(hist):
#     dist = {}
#     for gray in range(256):
#         value = 0.0
#         for k in range(256):
#             value += hist[k][0] * abs(gray - k)
#         dist[gray] = value
#     return dist
#
#
# def get_img_name(src):
#     return src.split('\\')[-1][:-4]
#
#
# def save_saliency_image(saliency_image, img_name):
#     save_dir = "\\"
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     cv2.imwrite(save_dir + "saliency_%s.jpg" % img_name, saliency_image*255)
#
#
# def plot(image):
#     cv2.imshow("gray saliency image", image)
#     cv2.waitKey(0)
#
#
# def main(file_path):
#     start = time.time()
#     image_gray = read_image(file_path)
#     image_name = get_img_name(file_path)
#     saliency_image = LC(image_gray)
#     print("111",saliency_image)
#     end = time.time()
#     print("Duration: %.2f seconds." % (end - start))
#
#     plot(saliency_image)
#     # save_saliency_image(saliency_image, image_name)
#
#
# if __name__ == "__main__":
#     # file = "SkinImages\\*.jpg"
#     # images_list = glob.glob(file)
#     # for img in images_list:
#     #     thread = Thread(target=main, args={img, })
#     #     thread.start()
#
#     file = "Datas/ad1.png"
#     main(file)
#
#
#
#
import os
from decimal import getcontext

# import cv2
# import numpy as np
#
# # 读取图像
# image = cv2.imread('ad1.png')
# # image = np.float64(image)
# # 转换图像为灰度
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 计算图像的梯度
# gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
# gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
#
# # 计算梯度幅值和方向
# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
# gradient_direction = np.arctan2(gradient_y, gradient_x)
#
# # 计算显著性图
# saliency_map = gradient_magnitude - gradient_direction
#
# # 显示原始图像和显著性图
# cv2.imshow('Original Image', image)
# cv2.imshow('Saliency Map', saliency_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
#
# # 加载预训练的VGG16模型（去除全连接层）
# base_model = VGG16(weights='imagenet', include_top=False)
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)
#
# def preprocess_image(img):
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array
#
# def compute_saliency_map(frame):
#     frame = cv2.resize(frame, (224, 224))
#     img_array = np.expand_dims(frame, axis=0)
#     img_array = preprocess_input(img_array)
#     feature_map = model.predict(img_array)
#     saliency_map = np.sum(np.abs(feature_map), axis=-1)[0]
#     saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
#     return saliency_map
#
# # 读取视频
# video_path = 'Datas/023.mp4'
# cap = cv2.VideoCapture(video_path)
#
# # 视频显著性区域累加
# total_saliency_map = None
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 计算显著性图
#     saliency_map = compute_saliency_map(frame)
#
#     # 显示显著性图
#     cv2.imshow('Saliency Map', (saliency_map * 255).astype(np.uint8))
#
#     # 累加显著性图
#     if total_saliency_map is None:
#         total_saliency_map = np.zeros_like(saliency_map, dtype=np.float32)
#
#     total_saliency_map += saliency_map.astype(np.float32)
#
#     # 按ESC键退出
#     if cv2.waitKey(30) & 0xFF == 27:
#         break
#
# # 归一化总显著性图
# total_saliency_map /= total_saliency_map.max()
#
# # 显示整体显著性图
# cv2.imshow('Total Saliency Map', (total_saliency_map * 255).astype(np.uint8))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
getcontext().prec = 10  # 设置精度为10位
getcontext().rounding = "ROUND_HALF_UP"  # 设置舍入模式为四舍五入
env = dict()
x_list = []
y_list = []
folder_path = "Datas/VR_frame_50"
file_list = os.listdir(folder_path)
# print(os.path.join(folder_path, file_list[0]))
file_path = os.path.join(folder_path, file_list[1])
with open(file_path, newline='') as file:
    eye_data_text = file.readlines()
    for line in eye_data_text:
        eye_list = line.split(',')
        frame, forward_x, forward_y, eye_x, eye_y = int(eye_list[1]), float(eye_list[3]), float(
            eye_list[4]), float(eye_list[6]), float(eye_list[7])
        env[frame] = {'frame': frame, 'forward_x': forward_x, 'forward_y': forward_y, 'eye_x': eye_x,
                      'eye_y': eye_x}
        x_list.append([float(eye_list[6]),float(eye_list[7])])
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
normalized_heat_map = (heat_map_smoothed - heat_map_smoothed.min()) / (heat_map_smoothed.max() - heat_map_smoothed.min())

# 自适应阈值设置
adaptive_threshold = np.mean(normalized_heat_map) + np.std(normalized_heat_map)
ad_width = 0.01
ad_heigth = 0.05
ad_state_x = 0.444
ad_state_y = 0.49
# 定义一个矩形区域 [y_min, y_max, x_min, x_max]，已归一化
rectangle_bbox_normalized = [0.49-0.05/2, 0.49+0.05/2, 0.444-0.01/2, 0.444+0.01/2]

# 转换矩形区域坐标为图像坐标
rectangle_bbox_image = [
    int(rectangle_bbox_normalized[0] * image_size[0]),
    int(rectangle_bbox_normalized[1] * image_size[0]),
    int(rectangle_bbox_normalized[2] * image_size[1]),
    int(rectangle_bbox_normalized[3] * image_size[1])
]

# 提取矩形区域内的显著性信息
rectangle_heat_map = normalized_heat_map[rectangle_bbox_image[1]:rectangle_bbox_image[0], rectangle_bbox_image[2]:rectangle_bbox_image[3]]

# 判断是否有重合区域
overlap = np.any(rectangle_heat_map > adaptive_threshold)

# 显示结果
cv2.imshow('Normalized Heat Map', (normalized_heat_map * 255).astype(np.uint8))
cv2.rectangle(normalized_heat_map, (rectangle_bbox_image[2], rectangle_bbox_image[1]), (rectangle_bbox_image[3], rectangle_bbox_image[0]), (255, 255, 255), 2)

if overlap:
    print("矩形区域与显著性区域有重合")
else:
    print("矩形区域与显著性区域无重合")

cv2.imshow('Rectangle Region', normalized_heat_map)
cv2.waitKey(0)
cv2.destroyAllWindows()


