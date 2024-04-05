#
#
#
# from PIL import Image, ImageDraw
#
#
# def draw_rectangle(image_path, output_path, normalized_bottom_left, normalized_top_right, color=(255, 0, 0),
#                    thickness=8):
#     """
#     在图片指定归一化区域内画一个矩形框并保存结果。
#
#     Parameters:
#         - image_path: 输入图片路径
#         - output_path: 输出图片路径
#         - normalized_bottom_left: 归一化矩形框左下角坐标 (x, y)
#         - normalized_top_right: 归一化矩形框右上角坐标 (x, y)
#         - color: 矩形框颜色，默认为红色 (R, G, B)
#         - thickness: 矩形框线条粗细，默认为2
#     """
#     # 打开图片
#     image = Image.open(image_path)
#
#     # 获取图片宽度和高度
#     width, height = image.size
#
#     # 归一化坐标
#     pixel_bottom_left = (int(normalized_bottom_left[0] * width), int((1 - normalized_bottom_left[1]) * height))
#     pixel_top_right = (int(normalized_top_right[0] * width), int((1 - normalized_top_right[1]) * height))
#
#     # 创建绘图对象
#     draw = ImageDraw.Draw(image)
#
#     # 画矩形框
#     draw.rectangle([pixel_bottom_left, pixel_top_right], outline=color, width=thickness)
#
#     # 保存结果
#     image.save(output_path)
#
#
# # 例子：画一个矩形框并保存结果
# input_image_path = "ad1.png"
# output_image_path = "output_image_with_rectangle.jpg"
# normalized_bottom_left = (0.43, 0.58)
# normalized_top_right = (0.46, 0.35)
#
# draw_rectangle(input_image_path, output_image_path, normalized_bottom_left, normalized_top_right)
import os

# from PIL import Image, ImageDraw
#
# def draw_rectangles(image_path, output_path, normalized_bottom_left1, normalized_top_right1,
#                      normalized_bottom_left2, normalized_top_right2, color=(255, 0, 0), thickness=8):
#     """
#     在图片指定归一化区域内画两个矩形框并保存结果。
#
#     Parameters:
#         - image_path: 输入图片路径
#         - output_path: 输出图片路径
#         - normalized_bottom_left1: 第一个归一化矩形框左下角坐标 (x, y)
#         - normalized_top_right1: 第一个归一化矩形框右上角坐标 (x, y)
#         - normalized_bottom_left2: 第二个归一化矩形框左下角坐标 (x, y)
#         - normalized_top_right2: 第二个归一化矩形框右上角坐标 (x, y)
#         - color: 矩形框颜色，默认为红色 (R, G, B)
#         - thickness: 矩形框线条粗细，默认为2
#     """
#     # 打开图片
#     image = Image.open(image_path)
#
#     # 获取图片宽度和高度
#     width, height = image.size
#
#     # 归一化坐标
#     pixel_bottom_left1 = (int(normalized_bottom_left1[0] * width), int((1 - normalized_bottom_left1[1]) * height))
#     pixel_top_right1 = (int(normalized_top_right1[0] * width), int((1 - normalized_top_right1[1]) * height))
#
#     pixel_bottom_left2 = (int(normalized_bottom_left2[0] * width), int((1 - normalized_bottom_left2[1]) * height))
#     pixel_top_right2 = (int(normalized_top_right2[0] * width), int((1 - normalized_top_right2[1]) * height))
#
#     # 创建绘图对象
#     draw = ImageDraw.Draw(image)
#
#     # 画矩形框1
#     draw.rectangle([pixel_bottom_left1, pixel_top_right1], outline=color, width=thickness)
#
#     # 画矩形框2
#     draw.rectangle([pixel_bottom_left2, pixel_top_right2], outline=color, width=thickness)
#
#     # 保存结果
#     image.save(output_path)
#
# # 例子：画两个矩形框并保存结果
# input_image_path = "Datas/Testing_scenarios/ad2.png"
# output_image_path = "output_image_with_rectangles.jpg"
# normalized_bottom_left1 = (0.4, 0.6)
# normalized_top_right1 = (0.46, 0.35)
# normalized_bottom_left2 = (0.44, 0.5)
# normalized_top_right2 = (0.45, 0.4)
#
# draw_rectangles(input_image_path, output_image_path, normalized_bottom_left1, normalized_top_right1,
#                 normalized_bottom_left2, normalized_top_right2)
# image_path = "0000.png"
# output_path = "output_image.jpg"
# image = Image.open(image_path)
# color = (255, 0, 0)
# thickness = 8
# # 获取图片宽度和高度
# for i in range(10):
#     width, height = image.size
#     normalized_bottom_left1 = (0.43+0.01*i, 0.58)
#     normalized_top_right1 = (0.46+0.01*i, 0.35)
#     # 归一化坐标
#     pixel_bottom_left1 = (int(normalized_bottom_left1[0] * width), int((1 - normalized_bottom_left1[1]) * height))
#     pixel_top_right1 = (int(normalized_top_right1[0] * width), int((1 - normalized_top_right1[1]) * height))
#
#
#     # 创建绘图对象
#     draw = ImageDraw.Draw(image)
#
#     # 画矩形框1
#     draw.rectangle([pixel_bottom_left1, pixel_top_right1], outline=color, width=thickness)
# image.save(output_path)

from PIL import Image, ImageDraw


# ad_limit_x = 0.282
# ad_limit_y = 0.4
# ad_limit_width = 0.05
# ad_limit_height = 0.14
#
# ad_state_x = 0.282
# ad_state_y = 0.4
# ad_width = 0.025
# ad_heigth = 0.07
# image_path = "Datas/Testing_scenarios/ad7.png"
# output_path = "test_images/0000_test_7.jpg"
# image = Image.open(image_path)
# color = (255, 0, 0)
# color1=(0,255,0)
# thickness = 8
# width, height = image.size
# normalized_bottom_left1 = (ad_limit_x-(ad_limit_width/2), ad_limit_y+(ad_limit_height/2))
# normalized_top_right1 = (ad_limit_x+ad_limit_width/2, ad_limit_y-ad_limit_height/2)
# normalized_bottom_left2=(ad_state_x-ad_width/2, ad_state_y+ad_heigth/2)
# normalized_top_right2=(ad_state_x+ad_width/2,ad_state_y-ad_heigth/2)
# # 归一化坐标
# pixel_bottom_left1 = (int(normalized_bottom_left1[0] * width), int((1 - normalized_bottom_left1[1]) * height))
# pixel_top_right1 = (int(normalized_top_right1[0] * width), int((1 - normalized_top_right1[1]) * height))
# pixel_bottom_left2 = (int(normalized_bottom_left2[0] * width), int((1 - normalized_bottom_left2[1]) * height))
# pixel_top_right2 = (int(normalized_top_right2[0] * width), int((1 - normalized_top_right2[1]) * height))
# # 创建绘图对象
# draw = ImageDraw.Draw(image)
#
# # 画矩形框1
# draw.rectangle([pixel_bottom_left1, pixel_top_right1], outline=color, width=thickness)
# draw.rectangle([pixel_bottom_left2, pixel_top_right2], outline=color, width=thickness)
#
# image.save(output_path)


folder_path_1 = 'Datas/Testing_scenarios'
folder_path_2 = 'Datas/Ad/ywh'

# 获取两个文件夹中所有图片文件的名称
images_1 = [f for f in os.listdir(folder_path_1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
images_2 = [f for f in os.listdir(folder_path_2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
ad_xy_list=[[0.45,0.455,0.02,0.08],[0.44,0.45,0.04,0.15],[0.425,0.53,0.01,0.05],[0.43,0.5,0.04,0.053],[0.12,0.54,0.03,0.08],[0.55,0.5,0.07,0.21],[0.282,0.4,0.0255,0.0714]]
i=0
# 确保列表长度一致
if len(images_1) == len(images_2):
    # 同时遍历两个列表
    for filename_1, filename_2 in zip(images_1, images_2):
        # 构建完整路径
        image_path_1 = os.path.join(folder_path_1, filename_1)
        image_path_2 = os.path.join(folder_path_2, filename_2)
        background_path = image_path_1

        # 小图片路径
        overlay_path = image_path_2

        # 小图片位置，坐标归一化
        overlay_center = (ad_xy_list[i][0],ad_xy_list[i][1])  # (x, y)
        # 小图片大小，坐标归一化
        overlay_size = (ad_xy_list[i][2], ad_xy_list[i][3])  # (width, height)

        background = Image.open(background_path)
        overlay = Image.open(overlay_path)

        # 获取大图片的宽度和高度
        bg_width, bg_height = background.size

        # 归一化处理，转换为实际坐标和大小
        overlay_center_x, overlay_center_y = overlay_center
        overlay_center_x = int(overlay_center_x * bg_width)
        overlay_center_y = int(overlay_center_y * bg_height)

        overlay_width, overlay_height = overlay_size
        overlay_width = int(overlay_width * bg_width)
        overlay_height = int(overlay_height * bg_height)

        # 计算小图片左下角坐标
        overlay_x = overlay_center_x - overlay_width // 2
        overlay_y = overlay_center_y - overlay_height // 2

        # 调整小图片大小
        overlay = overlay.resize((overlay_width, overlay_height), Image.Resampling.LANCZOS)

        # 将小图片粘贴到大图片的指定位置
        background.paste(overlay, (overlay_x, bg_height - overlay_y - overlay_height))

        # 保存结果
        background.save(f"Datas/Experiment_images/ywh/ywh_ad_{i+1}.png")
        i=i+1

else:
    print("Folders do not contain the same number of images.")

