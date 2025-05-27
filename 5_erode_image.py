import cv2
import os
import numpy as np
import shutil
# 输入文件夹路径
input_folder = './results/anomaly_images_thresholded/fruit_jelly/test_private_mixed'
# 输出文件夹路径
output_folder = './results/anomaly_images_thresholded/fruit_jelly/test_private_mixed_new'
os.makedirs(output_folder, exist_ok=True)

# 腐蚀的核大小
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# 遍历文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 执行腐蚀操作
        eroded = cv2.erode(img, kernel, iterations=1)

        # 保存腐蚀后的图像
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, eroded)

print("腐蚀操作完成，结果已保存。")


# 输入文件夹路径
input_folder = './results/anomaly_images_thresholded/fruit_jelly/test_private'
# 输出文件夹路径
output_folder = './results/anomaly_images_thresholded/fruit_jelly/test_private_new'
os.makedirs(output_folder, exist_ok=True)

# 腐蚀的核大小
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# 遍历文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 执行腐蚀操作
        eroded = cv2.erode(img, kernel, iterations=1)

        # 保存腐蚀后的图像
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, eroded)

print("腐蚀操作完成，结果已保存。")


# 你的根文件夹路径
root_dir = './results/anomaly_images_thresholded/fruit_jelly'  # 请修改为你的实际路径

# 需要删除的文件夹名
folders_to_delete = ['test_private', 'test_private_mixed']

# 需要重命名的文件夹（原名 -> 新名）
folders_to_rename = {
    'test_private_mixed_new': 'test_private_mixed',
    'test_private_new': 'test_private',
}

# 1. 删除指定文件夹
for folder in folders_to_delete:
    folder_path = os.path.join(root_dir, folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"已删除文件夹: {folder_path}")

# 2. 重命名merge文件夹
for old_name, new_name in folders_to_rename.items():
    old_path = os.path.join(root_dir, old_name)
    new_path = os.path.join(root_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"已重命名: {old_path} -> {new_path}")

print("操作完成。")

