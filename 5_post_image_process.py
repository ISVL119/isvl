import cv2
import numpy as np
import os
from tqdm import tqdm  # 可选：显示处理进度条

# 设置输入和输出文件夹路径
input_folder = r'./results/anomaly_images_thresholded/fabric/test_private_mixed'
output_folder = r"./results/anomaly_images_thresholded/fabric/test_private_mixed_new"
os.makedirs(output_folder, exist_ok=True)

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

for file_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 闭运算填补小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 填充封闭区域
    h, w = closed.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    floodfilled = closed.copy()
    cv2.floodFill(floodfilled, mask, (0, 0), 255)
    floodfilled_inv = cv2.bitwise_not(floodfilled)
    filled_image = cv2.bitwise_or(closed, floodfilled_inv)

    # 判断填补后是否为95%以上白色
    white_pixel_ratio = np.sum(filled_image == 255) / (h * w)
    if white_pixel_ratio > 0.95:
        # 如果几乎全白，保存原始图片
        save_image = image
    else:
        save_image = filled_image

    # 保存图像到输出路径
    save_path = os.path.join(output_folder, file_name)
    cv2.imwrite(save_path, save_image)

print(f"全部处理完成，保存到文件夹: {output_folder}")


input_folder = r'./results/anomaly_images_thresholded/fabric/test_private'
output_folder = r"./results/anomaly_images_thresholded/fabric/test_private_new"
os.makedirs(output_folder, exist_ok=True)

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

for file_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 闭运算填补小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 填充封闭区域
    h, w = closed.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    floodfilled = closed.copy()
    cv2.floodFill(floodfilled, mask, (0, 0), 255)
    floodfilled_inv = cv2.bitwise_not(floodfilled)
    filled_image = cv2.bitwise_or(closed, floodfilled_inv)

    # 判断填补后是否为95%以上白色
    white_pixel_ratio = np.sum(filled_image == 255) / (h * w)
    if white_pixel_ratio > 0.95:
        # 如果几乎全白，保存原始图片
        save_image = image
    else:
        save_image = filled_image

    # 保存图像到输出路径
    save_path = os.path.join(output_folder, file_name)
    cv2.imwrite(save_path, save_image)

print(f"全部处理完成，保存到文件夹: {output_folder}")
