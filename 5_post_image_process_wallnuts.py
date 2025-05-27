import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shutil

def extract_foreground_by_hsv(img_bgr, k=2, dilation_kernel_size=15):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    data = hsv.reshape((-1, 3))
    hv_data = data[:, [0, 2]]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(hv_data)
    labels = kmeans.labels_.reshape((h, w))
    cluster_means = [np.mean(hv_data[labels.reshape(-1) == i, 1]) for i in range(k)]
    bg_label = np.argmin(cluster_means)
    foreground_mask = (labels != bg_label).astype(np.uint8) * 255
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)
    return dilated_foreground_mask

def filter_anomaly_with_foreground_mask(anomaly_map, foreground_mask):
    return cv2.bitwise_and(anomaly_map, foreground_mask)

def process_folder(image_dir, anomaly_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        anomaly_path = os.path.join(anomaly_dir, filename)

        img = cv2.imread(img_path)
        anomaly = cv2.imread(anomaly_path, cv2.IMREAD_GRAYSCALE)

        if img is None or anomaly is None:
            print(f"跳过无效文件: {filename}")
            continue

        fg_mask = extract_foreground_by_hsv(img)
        anomaly_filtered = filter_anomaly_with_foreground_mask(anomaly, fg_mask)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, anomaly_filtered)

    print("批量处理完成。")

# 示例用法路径（请替换为你的实际路径）
image_dir = './mvtec_ad_2/walnuts/test_private_mixed'
anomaly_dir = './results/anomaly_images_thresholded/walnuts/test_private_mixed' 
output_dir = './results/anomaly_images_thresholded/walnuts/test_private_mixed_new'
process_folder(image_dir, anomaly_dir, output_dir)


# 你的根文件夹路径
root_dir = './results/anomaly_images_thresholded/walnuts'  # 请修改为你的实际路径

# 需要删除的文件夹名
folders_to_delete = ['test_private_mixed']

# 需要重命名的文件夹（原名 -> 新名）
folders_to_rename = {
    'test_private_mixed_new': 'test_private_mixed'
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



from tqdm import tqdm
input_folder = r'./results/anomaly_images_thresholded/walnuts/test_private_mixed'
output_folder = r"./results/anomaly_images_thresholded/walnuts/test_private_mixed_new"
os.makedirs(output_folder, exist_ok=True)

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

for file_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 闭运算填补小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
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


input_folder = r'./results/anomaly_images_thresholded/walnuts/test_private'
output_folder = r"./results/anomaly_images_thresholded/walnuts/test_private_new"
os.makedirs(output_folder, exist_ok=True)

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

for file_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 闭运算填补小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
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


# 你的根文件夹路径
root_dir = './results/anomaly_images_thresholded/walnuts'  # 请修改为你的实际路径

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