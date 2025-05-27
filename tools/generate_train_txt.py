import os

# 设置文件夹路径
folder_path = "/home/hy/xsy_pan/250411CPRmaste/sythesized_fruit"
output_txt_path = os.path.join(folder_path, "train.txt")

# 获取所有不含 "mask" 的图片文件名
file_list = [f for f in os.listdir(folder_path) if f.endswith('.png') and 'mask' not in f]

# 按文件名排序（可选）
file_list.sort()

# 生成每一行的内容
lines = []
for filename in file_list:
    # 提取前缀：去掉 _syn_后面的部分
    # 例如 "000_regular_syn_0.png" -> "000_regular"
    prefix = filename.split("_syn_")[0]
    reference_path = f"train/good/{prefix}.png"
    lines.append(f"{filename} {reference_path}")

# 写入 train.txt
with open(output_txt_path, "w") as f:
    f.write("\n".join(lines))

print(f"train.txt 文件已生成，共包含 {len(lines)} 行。")
