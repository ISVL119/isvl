import os
import shutil

# 你的根文件夹路径
root_dir = './results/anomaly_images_thresholded/fabric'  # 请修改为你的实际路径

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
