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
#     file = "Datas/0000.png"
#     main(file)
#
#
#
#

import cv2
import numpy as np

# 读取图像
image = cv2.imread('0000.png')
# image = np.float64(image)
# 转换图像为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算图像的梯度
gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# 计算梯度幅值和方向
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# 计算显著性图
saliency_map = gradient_magnitude - gradient_direction

# 显示原始图像和显著性图
cv2.imshow('Original Image', image)
cv2.imshow('Saliency Map', saliency_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
