import os
import glob
import random
import numpy as np
from PIL import Image, ImageOps
import cv2
import argparse
from pathlib import Path
import torch

# 切换到项目根目录（假设脚本在 tools/ 下）
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)


def parse_args():
    p = argparse.ArgumentParser(
        description="generate synthesized data by hand with normal images, anomalous patches and foreground"
    )
    p.add_argument("--anomaly_dir",default='data/anomaly', type=str,
                   help="path for anomaly pathces")
    p.add_argument("--normal_dir", default='data/mvtec_ad_2_vial_fruit', type=str,
                   help="normal background images")
    p.add_argument("--mask_dir", type=str, default='log/foreground/foreground_mvtec_ad_2_vial_fruit',
                   help="foreground masks for normal images")
    p.add_argument("--output_dir", default=None, type=str,
                   help="path for output sythesized images")
    p.add_argument("--dataset-name", type=str, default="mvtec_ad_2_vial_fruit")
    p.add_argument("--num_per_image", type=int, default=5)
    p.add_argument("--resize", default=640, type=int,
                   help="resize 正常图与 mask 到的尺寸（例如 640）")
    p.add_argument("--seed", type=int, default=6,
                   help="随机种子，确保可复现")
    p.add_argument("--category", type=str, default=None,
                   help="只对其中一类做")
    return p.parse_args()

def seed_all(seed: int):
    # Python random
    random.seed(seed)                      # seed Python RNG :contentReference[oaicite:0]{index=0}
    # NumPy
    np.random.seed(seed)                   # seed NumPy RNG :contentReference[oaicite:1]{index=1}
    # PyTorch CPU
    torch.manual_seed(seed)                # seed PyTorch CPU RNG :contentReference[oaicite:2]{index=2}
    # PyTorch CUDA (若有)
    torch.cuda.manual_seed(seed)           # seed 单 GPU :contentReference[oaicite:3]{index=3}
    torch.cuda.manual_seed_all(seed)       # seed 多 GPU :contentReference[oaicite:4]{index=4}
    # 禁止 CuDNN 的非确定性算法，以提高可复现性
    torch.backends.cudnn.deterministic = True   # 保证确定性 :contentReference[oaicite:5]{index=5}
    torch.backends.cudnn.benchmark     = False  # 禁用基准测试模


def random_anomaly_patch(anom_img: Image.Image,  min_scale=0.005, max_scale=0.01):
    """从一张异常图像随机缩放、旋转，并裁剪一个矩形补丁。"""
    # 随机缩放
    w, h = anom_img.size
    scale = random.uniform(min_scale, max_scale)
    new_w, new_h = int(w * scale), int(h * scale)
    # max_patch_size = 80  # 根据需要设置最大尺寸
    # new_w = min(int(w * scale), max_patch_size)
    # new_h = min(int(h * scale), max_patch_size)
    patch = anom_img.resize((new_w, new_h), Image.NEAREST)

    # 随机旋转
    angle = random.uniform(0, 360)
    patch = patch.rotate(angle,  resample=Image.Resampling.NEAREST,expand=True)


    return patch

def overlay_patch_on_image(base_img: Image.Image,
                           fg_mask: np.ndarray,
                           patch: Image.Image):
    """
    在 base_img 的前景区域随机位置叠加 patch，
    并准确返回：
      1) 合成图 (PIL.Image)
      2) 有效叠加掩码 valid_mask (np.ndarray, 0 或 1，shape 与 base_img 一致)
    """
    bw, bh = base_img.size
    pw, ph = patch.size
    # 如果补丁比图大，跳过
    if pw > bw or ph > bh:
        return base_img, np.zeros((bh, bw), dtype=np.uint8)

    # 随机选前景点
    ys, xs = np.where(fg_mask >=0.8)
    if len(xs) == 0:
        return base_img, np.zeros((bh, bw), dtype=np.uint8)
    idx = random.randrange(len(xs))
    cx, cy = xs[idx], ys[idx]

    # 计算放置位置
    x1 = max(0, cx - pw//2)
    y1 = max(0, cy - ph//2)
    x2 = min(bw, x1 + pw)
    y2 = min(bh, y1 + ph)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return base_img, np.zeros((bh, bw), dtype=np.uint8)

    # 在 patch 上对应的裁剪坐标
    src_x = max(0, x1 - (cx - pw//2))
    src_y = max(0, y1 - (cy - ph//2))

    # 分离 RGBA 与 二值 alpha
    patch_np = np.array(patch)
    if patch.mode == 'RGBA':
        raw_alpha  = patch_np[:, :, 3]
        alpha_patch= (raw_alpha > 0).astype(float)     # 硬阈值二值化
        rgb_patch  = patch_np[:, :, :3].astype(float)
    else:
        gray       = cv2.cvtColor(patch_np, cv2.COLOR_BGR2GRAY)
        alpha_patch= (gray < 250).astype(float)
        rgb_patch  = patch_np.astype(float)

    # 裁剪补丁与 alpha 为实际叠加大小
    alpha_crop = alpha_patch[src_y:src_y+h, src_x:src_x+w]
    rgb_crop   = rgb_patch[src_y:src_y+h, src_x:src_x+w, :]

    # 从 fg_mask 中裁剪子掩码
    sub_mask   = fg_mask[y1:y1+h, x1:x1+w].astype(float)
    # 假设 sub_mask 原本是 float 数组，值在 [0,1] 范围内


    # —— 新增：确保两者 shape 一致 —— #
    # 找到最小的高和宽
    h_min = min(alpha_crop.shape[0], sub_mask.shape[0])
    w_min = min(alpha_crop.shape[1], sub_mask.shape[1])
    # 按最小尺寸裁剪
    alpha_crop = alpha_crop[:h_min, :w_min]
    sub_mask = sub_mask[:h_min, :w_min]
    rgb_crop = rgb_crop[:h_min, :w_min, :]

    # 方法一：布尔索引 + astype
    sub_mask_binary = np.where(sub_mask > 0.45, 1.0, 0.0)

    # 真实叠加区域：补丁不透明 且 前景掩码 =1
    valid_crop = (alpha_crop * sub_mask)  # 0/1
    valid_crop_mask = (alpha_crop * sub_mask_binary).astype(np.uint8)  # 0/1

    # 构造全图级 valid_mask，其他区域保留 0
    valid_mask = np.zeros((bh, bw), dtype=np.uint8)
    valid_mask[y1:y1+h_min, x1:x1+w_min] = valid_crop_mask

    # 合成图像：仅在 valid_crop=1 处用 patch，其他位置保持 base
    base_np = np.array(base_img).astype(float)
    for c in range(3):
        base_np[y1:y1+h_min, x1:x1+w_min, c] = (
            valid_crop * rgb_crop[:, :, c] +
            (1 - valid_crop) * base_np[y1:y1+h_min, x1:x1+w_min, c]
        )

    return Image.fromarray(base_np.astype(np.uint8)), valid_mask


def read_image(path, resize=None):
    """
    Read an image from the specified path and optionally resize it.

    Parameters:
        path (str): The path to the image file.
        resize (tuple, optional): Tuple containing the desired width and height to resize the image to.

    Returns:
        numpy.ndarray: The image data as a NumPy array.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, dsize=resize)
    return img

import os

def generate_train_txt(folder_path):
    # folder_path is a Path to .../fruit_jelly
    category = folder_path.name
    output_txt_path = folder_path / "train.txt"

    # collect all synthetic PNGs without "mask"
    file_list = [
        f for f in os.listdir(folder_path)
        if f.endswith('.png') and 'mask' not in f
    ]
    file_list.sort()

    lines = []
    for filename in file_list:
        # strip off "_syn_" and category_ prefix
        prefix = filename.split("_syn_")[0]
        prefix_key = f"{category}_"
        if prefix.startswith(prefix_key):
            prefix = prefix[len(prefix_key):]

        # left side: include category/filename
        left = f"{category}/{filename}"
        # right side: always train/good/<original>.png
        right = f"train/good/{prefix}.png"

        lines.append(f"{left} {right}")

    # write out
    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"train.txt generated at {output_txt_path}, {len(lines)} entries.")
    for line in lines[:5]:
        print("  ", line)



args = parse_args()

# —— 最早调用 —— #
# 1) Python 哈希
os.environ['PYTHONHASHSEED'] = str(args.seed)
# 2) 所有库的随机种子
seed_all(args.seed)
# 3) 强制 PyTorch 用确定性算法
torch.use_deterministic_algorithms(True)

# 切换工作目录
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)

normal_root  = Path(args.normal_dir)
anomaly_root = Path(args.anomaly_dir)
mask_root    = Path(args.mask_dir)

if args.output_dir is None:
    args.output_dir = f'log/synthesized/synthesized_{args.dataset_name}'

output_root  = Path(args.output_dir)
output_root.mkdir(parents=True, exist_ok=True)

# 确保目录顺序一致
all_cats = sorted([d for d in normal_root.iterdir() if d.is_dir()])

for idx, category_dir in enumerate(all_cats):
    cat_name = category_dir.name  # 'fruit_jelly'
    # 如果用户指定了 --category，那么只处理对应索引的那个
    if args.category is not None and cat_name != args.category:
        continue

    cat_name = category_dir.name
    category_out_dir = output_root / cat_name
    category_out_dir.mkdir(parents=True, exist_ok=True)
    # 正常图所在的子目录：<normal_dir>/<category>/train/good
    norm_good_dir = category_dir / "train" / "good"
    if not norm_good_dir.exists():
        print(f"[WARN] no normal train/good dir for {cat_name}, skipping")
        continue

    # 对应的前景掩码目录：<mask_dir>/<category>/train/good
    mask_good_dir = mask_root / cat_name / "train" / "good"
    if not mask_good_dir.exists():
        print(f"[WARN] no mask train/good dir for {cat_name}, skipping")
        continue

    # 异常补丁根目录：<anomaly_dir>/<category>
    anomaly_cat_dir = anomaly_root / cat_name
    if not anomaly_cat_dir.exists():
        print(f"[WARN] no anomaly dir for {cat_name}, skipping")
        continue
    category_out_dir = output_root / cat_name
    category_out_dir.mkdir(parents=True, exist_ok=True)
    # 遍历所有正常图像
    for norm_path in norm_good_dir.glob('*.[pj][pn]g'):
        norm_name = norm_path.stem

        # 读取 & resize 正常图
        img = Image.open(norm_path).convert('RGB') \
                 .resize((args.resize, args.resize), Image.BICUBIC)
        # 读取 & resize 对应前景掩码（.npy）
        mask_np = np.load(mask_good_dir / f"f_{norm_name}.npy")
        mask = cv2.resize(mask_np, (args.resize, args.resize),
                          interpolation=cv2.INTER_NEAREST)

        # 找到当前类别下所有异常子类别目录列表
        sub_dirs = [d for d in anomaly_cat_dir.iterdir() if d.is_dir()]

        # 对这张正常图，针对每个异常子类别分别做合成
        for chosen in sub_dirs:
            # chosen.name 是子类别名，比如 'bottle'
            for i in range(args.num_per_image):
                # 从当前子类别目录随机选一张补丁
                patch_path = random.choice(list(chosen.glob('*.png')))
                patch = Image.open(patch_path).convert('RGBA')
                if chosen.name == 'bottle' or chosen.name == 'capsules' :
                    patch = random_anomaly_patch(patch, min_scale=0.1, max_scale=1.0)
                elif chosen.name == 'contamination':
                    patch = random_anomaly_patch(patch, min_scale=0.1, max_scale=0.8)
                elif chosen.name == 'tube':
                    patch = random_anomaly_patch(patch, min_scale=0.05, max_scale=0.3)
                elif chosen.name == 'hair':
                    patch = random_anomaly_patch(patch, min_scale=0.05, max_scale=0.2)
                elif chosen.name == 'screw' :
                    patch = random_anomaly_patch(patch, min_scale=0.05, max_scale=0.1)
                elif chosen.name == 'screw1':
                    patch = random_anomaly_patch(patch, min_scale=0.05, max_scale=0.2)
                else:
                    break
                # 叠加并保存
                out_img, valid_mask = overlay_patch_on_image(img, mask, patch)
                out_name = f"{cat_name}_{norm_name}_syn_{chosen.name}_{i}.png"
                out_img.save(output_root / cat_name / out_name)

                # 生成并保存二值 mask
                bin_mask = (valid_mask * 255).astype(np.uint8)
                mask_name = f"{cat_name}_{norm_name}_syn_{chosen.name}_{i}_mask.png"
                Image.fromarray(bin_mask).save(output_root / cat_name / mask_name)

    generate_train_txt(output_root / cat_name)

print("合成完成，结果保存在：", output_root)
