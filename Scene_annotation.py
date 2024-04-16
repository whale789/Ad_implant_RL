import cv2
import numpy as np

def draw_grid(img, grid_size, color, thickness, transparency):
    """在图像上绘制半透明的网格线"""
    overlay = img.copy()
    for x in range(0, img.shape[1], grid_size):
        cv2.line(overlay, (x, 0), (x, img.shape[0]), color, thickness)
    for y in range(0, img.shape[0], grid_size):
        cv2.line(overlay, (0, y), (img.shape[1], y), color, thickness)
    cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)
    return img

# 鼠标回调函数用于绘制矩形
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, image, clone

    # 转换坐标以匹配显示尺寸
    x_scaled, y_scaled = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x_scaled, y_scaled

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_clone = clone.copy()
            cv2.rectangle(temp_clone, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', temp_clone)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x_scaled, y_scaled), (0, 255, 0), 2)
        rectangles.append((ix, iy, x_scaled, y_scaled))

# 加载图片并调整大小
image_path = 'Datas/Testing_scenarios/ad2.png'  # 替换为你的图片路径
image = cv2.imread(image_path)
original_size = image.shape[1], image.shape[0]  # 保存原始尺寸
scaling_factor = 0.2  # 可根据需要调整缩放因子
image = cv2.resize(image, (int(original_size[0] * scaling_factor), int(original_size[1] * scaling_factor)))
clone = image.copy()
rectangles = []

# 初始化变量
ix, iy = -1, -1
drawing = False

# 网格大小
grid_size = 15
# 调用绘制网格的函数
image = draw_grid(image, grid_size, (255, 255, 255), 1, 0.5)

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('image', draw_rectangle)

# 主循环
while True:
    cv2.imshow('image', clone if drawing else image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# 输出标记的矩形
cv2.destroyAllWindows()
for rect in rectangles:
    print(f"Top-left: ({rect[0]}, {rect[1]}), Bottom-right: ({rect[2]}, {rect[3]})")
