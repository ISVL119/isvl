import os
import shutil

# 原始数据集根目录
src_root = "./mvtec_ad_2"
# 目标目录
dst_root = "./data/mvtec_test_vial_fruit"

# 类别名称
categories = ["vial", "fruit_jelly"]
# 要拷贝到 test 中的文件夹名
test_sources = ["test_private", "test_private_mixed"]

for category in categories:
    src_category_path = os.path.join(src_root, category)
    dst_category_path = os.path.join(dst_root, category)

    # 创建目标类目录及其子目录 train 和 test
    os.makedirs(os.path.join(dst_category_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_category_path, "test"), exist_ok=True)

    # 拷贝 train
    src_train_path = os.path.join(src_category_path, "train")
    dst_train_path = os.path.join(dst_category_path, "train")
    if os.path.exists(src_train_path):
        shutil.copytree(src_train_path, dst_train_path, dirs_exist_ok=True)

    # 拷贝 test_private 和 test_private_mixed 到 test/<same_name> 目录
    for test_folder in test_sources:
        src_test_path = os.path.join(src_category_path, test_folder)
        dst_test_subfolder_path = os.path.join(dst_category_path, "test", test_folder)

        if os.path.exists(src_test_path):
            shutil.copytree(src_test_path, dst_test_subfolder_path, dirs_exist_ok=True)

print("数据结构拷贝完成 ✅")
