import os
import cv2

# 阈值配置：每个类别可以有不同的子文件夹阈值，也可以有默认阈值（用"default"键）
thresholds = {
    "can": {
        "test_private": 79,
        "test_private_mixed": 96,
    },
    "fabric": {
        "test_private": 69,
        "test_private_mixed": 69,
    },
    "fruit_jelly": {
        "test_private": 254,
        "test_private_mixed": 254
    },
    "rice": {
        "test_private": 54,
        "test_private_mixed": 54
    },
    "sheet_metal": {
        "test_private": 63,
        "test_private_mixed":65
    },
    "vial": {
        "test_private": 164,
        "test_private_mixed": 177
    },
    "wallplugs": {
        "test_private": 99,
        "test_private_mixed": 108
    },
    "walnuts": {
        "test_private": 67,
        "test_private_mixed": 71
    }
}

def threshold_and_save_images_recursive(input_dir, output_dir):
    """
    递归地将input_dir下的所有单通道图片按类别+子文件夹对应的阈值进行二值化，
    并保留原目录结构保存到output_dir，统一保存为PNG格式。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)

                # 计算相对路径
                rel_path = os.path.relpath(file_path, input_dir)
                rel_parts = rel_path.split(os.sep)

                # 确保至少有类别和子目录
                if len(rel_parts) < 2:
                    print(f"Warning: File {file_path} not in expected category/subfolder structure.")
                    continue

                category = rel_parts[0]
                subfolder = rel_parts[1]

                # 获取阈值
                if category in thresholds:
                    category_thresholds = thresholds[category]
                    if subfolder in category_thresholds:
                        thresh = category_thresholds[subfolder]
                    elif "default" in category_thresholds:
                        thresh = category_thresholds["default"]
                        print(f"Info: Using default threshold for {category}/{subfolder}")
                    else:
                        print(f"Warning: No threshold found for {category}/{subfolder}, skipping.")
                        continue
                else:
                    print(f"Warning: Category '{category}' not in thresholds, skipping {file_path}")
                    continue

                # 读取图像
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to read image: {file_path}")
                    continue

                # 阈值处理
                _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

                # 保存路径
                base_filename = os.path.splitext(file)[0] + ".png"
                save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
                save_path = os.path.join(save_dir, base_filename)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                cv2.imwrite(save_path, binary_img)

if __name__ == "__main__":
    input_folder = "./results/anomaly_images"  # TODO: 替换为你的输入路径
    output_folder = "./results/anomaly_images_thresholded"  # TODO: 替换为你的输出路径

    threshold_and_save_images_recursive(input_folder, output_folder)
