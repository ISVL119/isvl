import os
import re
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

def group_images_by_prefix_recursive(root_dir):
    """递归查找所有图片并按规则分组"""
    # 支持 _split_、_gridNxM_row_col、_longedgeN_idx、和简单 _row_col
    pattern_split = re.compile(r"(.+)_split_(\d+)\.(\w+)$")
    pattern_grid = re.compile(r"(.+)_grid(\d+)x(\d+)_(\d+)_(\d+)\.(\w+)$")
    pattern_longedge = re.compile(r"(.+)_longedge(\d+)_(\d+)\.(\w+)$")
    pattern_simple_grid = re.compile(r"(.+)_([0-9]+)_([0-9]+)\.(\w+)$")

    split_groups = defaultdict(list)
    grid_groups = defaultdict(list)
    longedge_groups = defaultdict(list)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not is_image_file(file):
                continue

            abs_path = os.path.join(dirpath, file)
            rel_path = os.path.relpath(abs_path, root_dir)
            rel_dir = os.path.dirname(rel_path)

            match_split = pattern_split.match(file)
            match_grid = pattern_grid.match(file)
            match_longedge = pattern_longedge.match(file)
            match_simple_grid = pattern_simple_grid.match(file)

            if match_grid:
                base = match_grid.group(1)
                grid_w = int(match_grid.group(2))
                grid_h = int(match_grid.group(3))
                row = int(match_grid.group(4))
                col = int(match_grid.group(5))
                key = (rel_dir, base, f"grid{grid_w}x{grid_h}")
                grid_groups[key].append((row, col, abs_path))
            elif match_longedge:
                base = match_longedge.group(1)
                seg = int(match_longedge.group(2))
                idx = int(match_longedge.group(3))
                key = (rel_dir, base, f"longedge{seg}")
                longedge_groups[key].append((idx, abs_path))
            elif match_simple_grid:
                base = match_simple_grid.group(1)
                row = int(match_simple_grid.group(2))
                col = int(match_simple_grid.group(3))
                key = (rel_dir, base, "simple_grid")
                grid_groups[key].append((row, col, abs_path))
            elif match_split:
                base = match_split.group(1)
                idx = int(match_split.group(2))
                key = (rel_dir, base, "split")
                split_groups[key].append((idx, abs_path))

    return split_groups, grid_groups, longedge_groups

def merge_images_horizontally(image_paths, save_path):
    """横向拼接子图"""
    images = [Image.open(p) for _, p in sorted(image_paths)]
    mode = images[0].mode
    heights = [img.height for img in images]
    max_height = max(heights)
    total_width = sum(img.width for img in images)

    new_img = Image.new(mode, (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    new_img.save(save_path)

def merge_images_grid(image_tuples, save_path):
    """网格拼接子图"""
    sorted_images = sorted(image_tuples, key=lambda x: (x[0], x[1]))
    rows = max(r for r, c, _ in sorted_images) + 1
    cols = max(c for r, c, _ in sorted_images) + 1

    img_grid = [[None] * cols for _ in range(rows)]

    for row, col, path in sorted_images:
        img_grid[row][col] = Image.open(path)

    tile_w, tile_h = img_grid[0][0].size
    mode = img_grid[0][0].mode
    new_img = Image.new(mode, (tile_w * cols, tile_h * rows))

    for r in range(rows):
        for c in range(cols):
            if img_grid[r][c]:
                new_img.paste(img_grid[r][c], (c * tile_w, r * tile_h))

    new_img.save(save_path)

def reconstruct_images_recursive(input_dir, output_dir):
    split_groups, grid_groups, longedge_groups = group_images_by_prefix_recursive(input_dir)

    total = len(split_groups) + len(grid_groups) + len(longedge_groups)
    pbar = tqdm(total=total, desc="Reconstructing", ncols=100)

    # split类型横向拼接
    for (rel_dir, base, _), images in split_groups.items():
        ext = os.path.splitext(images[0][1])[1]
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        save_path = os.path.join(output_subdir, f"{base}{ext}")
        merge_images_horizontally(images, save_path)
        pbar.update(1)

    # grid类型网格拼接
    for (rel_dir, base, _), images in grid_groups.items():
        ext = os.path.splitext(images[0][2])[1]
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        save_path = os.path.join(output_subdir, f"{base}{ext}")
        merge_images_grid(images, save_path)
        pbar.update(1)

    # longedge类型横向拼接
    for (rel_dir, base, _), images in longedge_groups.items():
        ext = os.path.splitext(images[0][1])[1]
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        save_path = os.path.join(output_subdir, f"{base}{ext}")
        # 注意：longedge 只有idx，无row/col，直接横向拼接
        images_sorted = sorted(images, key=lambda x: x[0])
        merge_images_horizontally(images_sorted, save_path)
        pbar.update(1)

    pbar.close()
    print("拼接完成！")

# 示例使用
input_images= "./results/anomaly_images/can" 
output_images = "./results/anomaly_images/can_merge"
reconstruct_images_recursive(input_images, output_images)

input_images= "./results/anomaly_images/fabric" 
output_images = "./results/anomaly_images/fabric_merge"
reconstruct_images_recursive(input_images, output_images)

input_images= "./results/anomaly_images/rice" 
output_images = "./results/anomaly_images/rice_merge"
reconstruct_images_recursive(input_images, output_images)

input_images= "./results/anomaly_images/sheet_metal" 
output_images = "./results/anomaly_images/sheet_metal_merge"
reconstruct_images_recursive(input_images, output_images)

input_images= "./results/anomaly_images/walnuts" 
output_images = "./results/anomaly_images/walnuts_merge"
reconstruct_images_recursive(input_images, output_images)
