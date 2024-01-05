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
# input_image_path = "0000.png"
# output_image_path = "output_image_with_rectangle.jpg"
# normalized_bottom_left = (0.43, 0.58)
# normalized_top_right = (0.46, 0.35)
#
# draw_rectangle(input_image_path, output_image_path, normalized_bottom_left, normalized_top_right)

from PIL import Image, ImageDraw

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
# input_image_path = "0000.png"
# output_image_path = "output_image_with_rectangles.jpg"
# normalized_bottom_left1 = (0.43, 0.58)
# normalized_top_right1 = (0.46, 0.35)
# normalized_bottom_left2 = (0.44, 0.5)
# normalized_top_right2 = (0.45, 0.4)
#
# draw_rectangles(input_image_path, output_image_path, normalized_bottom_left1, normalized_top_right1,
#                 normalized_bottom_left2, normalized_top_right2)
image_path = "0000.png"
output_path = "output_image.jpg"
image = Image.open(image_path)
color = (255, 0, 0)
thickness = 8
# 获取图片宽度和高度
for i in range(10):
    width, height = image.size
    normalized_bottom_left1 = (0.43+0.01*i, 0.58)
    normalized_top_right1 = (0.46+0.01*i, 0.35)
    # 归一化坐标
    pixel_bottom_left1 = (int(normalized_bottom_left1[0] * width), int((1 - normalized_bottom_left1[1]) * height))
    pixel_top_right1 = (int(normalized_top_right1[0] * width), int((1 - normalized_top_right1[1]) * height))


    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 画矩形框1
    draw.rectangle([pixel_bottom_left1, pixel_top_right1], outline=color, width=thickness)
image.save(output_path)