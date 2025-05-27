import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

def copy_images_with_structure(src_dir, dst_dir):
    """
    递归复制src_dir下的所有图片到dst_dir，保持原有子目录结构
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            if is_image_file(file):  # 直接复用你已有的图片判断函数
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, start=src_dir)
                dst_path = os.path.join(dst_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)


def is_image_file(filename):
    """判断是否是图片文件"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    return filename.lower().endswith(valid_extensions)

def is_original_image(filename):
    """
    判断是不是原始图片（没有切分过的图片）。
    区分依据：不含 _split_，不以 _gridNxM_数字_数字 或 _longedgeN_数字 结尾。
    """
    name, ext = os.path.splitext(filename)
    if "_split_" in name or "longedge" in name or "grid" in name:
        return False
    # 匹配 "_数字_数字" 这样的后缀
    parts = name.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return False
    return True

def collect_image_paths(input_dir):
    """只收集原始图片路径"""
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if is_image_file(file) and is_original_image(file):
                image_paths.append(os.path.join(root, file))
    return image_paths

def save_sub_image(sub_img, save_dir, base_name, suffix, save_format, mode_tag=None):
    """保存子图像到指定目录，mode_tag 用于唯一标识不同切分方式"""
    if mode_tag:
        save_path = os.path.join(save_dir, f"{base_name}_{mode_tag}_{suffix}.{save_format.lower()}")
    else:
        save_path = os.path.join(save_dir, f"{base_name}_{suffix}.{save_format.lower()}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        if save_format.lower() in ('jpg', 'jpeg'):
            sub_img = sub_img.convert('RGB')  # jpg 不支持透明通道
        sub_img.save(save_path)

def split_image_grid(img, base_name, save_dir, grid_size, save_format):
    """按照网格方式切分图片"""
    img_width, img_height = img.size
    grid_w, grid_h = grid_size
    sub_img_width = img_width // grid_w
    sub_img_height = img_height // grid_h
    mode_tag = f"grid{grid_w}x{grid_h}"
    for row in range(grid_h):
        for col in range(grid_w):
            left = col * sub_img_width
            upper = row * sub_img_height
            right = (col + 1) * sub_img_width if col != grid_w - 1 else img_width
            lower = (row + 1) * sub_img_height if row != grid_h - 1 else img_height
            sub_img = img.crop((left, upper, right, lower))
            save_sub_image(sub_img, save_dir, base_name, f"{row}_{col}", save_format, mode_tag=mode_tag)

def split_image_long_edge(img, base_name, save_dir, num_splits, save_format):
    """按照长边均分切分图片"""
    img_width, img_height = img.size
    mode_tag = f"longedge{num_splits}"
    if img_width >= img_height:
        # 横图，按宽度切
        sub_img_width = img_width // num_splits
        for idx in range(num_splits):
            left = idx * sub_img_width
            right = (idx + 1) * sub_img_width if idx != num_splits - 1 else img_width
            sub_img = img.crop((left, 0, right, img_height))
            save_sub_image(sub_img, save_dir, base_name, f"{idx}", save_format, mode_tag=mode_tag)
    else:
        # 竖图，按高度切
        sub_img_height = img_height // num_splits
        for idx in range(num_splits):
            upper = idx * sub_img_height
            lower = (idx + 1) * sub_img_height if idx != num_splits - 1 else img_height
            sub_img = img.crop((0, upper, img_width, lower))
            save_sub_image(sub_img, save_dir, base_name, f"{idx}", save_format, mode_tag=mode_tag)

def split_image(
    image_path, 
    save_dir, 
    split_mode='grid', 
    grid_size=(2, 2), 
    num_splits=2, 
    save_format='png', 
    input_root_dir=None
):
    """根据模式拆分单张图片"""
    img = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if input_root_dir:
        relative_path = os.path.relpath(image_path, start=input_root_dir)
        relative_dir = os.path.dirname(relative_path)
        save_dir = os.path.join(save_dir, relative_dir)

    if split_mode == 'grid':
        split_image_grid(img, base_name, save_dir, grid_size, save_format)
    elif split_mode == 'long_edge':
        split_image_long_edge(img, base_name, save_dir, num_splits, save_format)
    else:
        raise ValueError(f"未知split_mode: {split_mode}，应为'grid'或'long_edge'")

def process_dataset(
    input_dir, 
    output_dir, 
    split_mode='grid', 
    grid_size=(2, 2), 
    num_splits=2, 
    save_format='png', 
    max_workers=8
):
    """批量处理整个数据集的所有图片，仅处理原始图片"""
    image_paths = collect_image_paths(input_dir)

    if not image_paths:
        print("未找到任何图片，请检查输入路径！")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        pbar = tqdm(total=len(image_paths), desc="Processing Images", ncols=100)

        for image_path in image_paths:
            futures.append(
                executor.submit(
                    split_image, 
                    image_path, 
                    output_dir, 
                    split_mode, 
                    grid_size, 
                    num_splits, 
                    save_format, 
                    input_dir
                )
            )

        for future in as_completed(futures):
            pbar.update(1)
        pbar.close()



if __name__ == "__main__":

    # can
    print("Processing can")
    root_dir = './mvtec_ad_2/can'
    save_dir = './mvtec_ad_2_aug/can'

    # train: 分别输出到 save_dir/train_long_edge 和 save_dir/train_grid
    train_dir = os.path.join(root_dir, 'train')

    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='long_edge',
        grid_size=(2,2),
        num_splits=2,
        save_format='png'
    )
    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='grid',
        grid_size=(4,2),
        num_splits=2,
        save_format='png'
    )

    # 其它子文件夹输出到 save_dir 下各自的_grid文件夹
    for sub in ['test_private', 'test_private_mixed', 'test_public', 'validation']:
        folder = os.path.join(root_dir, sub)
        grid_output = os.path.join(save_dir, sub)
        process_dataset(
            input_dir=folder,
            output_dir=grid_output,
            split_mode='grid',
            grid_size=(4,2),
            num_splits=2,
            save_format='png'
        )

    # fabric
    print("Processing fabric")
    root_dir = './mvtec_ad_2/fabric'
    save_dir = './mvtec_ad_2_aug/fabric'

    # train: 分别输出到 save_dir/train_long_edge 和 save_dir/train_grid
    train_dir = os.path.join(root_dir, 'train')

    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='grid',
        grid_size=(2,2),
        num_splits=2,
        save_format='png'
    )
    # 其它子文件夹输出到 save_dir 下各自的_grid文件夹
    for sub in ['test_private', 'test_private_mixed', 'test_public', 'validation']:
        folder = os.path.join(root_dir, sub)
        grid_output = os.path.join(save_dir, sub)
        process_dataset(
            input_dir=folder,
            output_dir=grid_output,
            split_mode='grid',
            grid_size=(2,2),
            num_splits=2,
            save_format='png'
        )
    
    copy_images_with_structure(
        src_dir='./mvtec_ad_2/fabric/train',
        dst_dir='./mvtec_ad_2_aug/fabric/train'
    )

    # fruit_jelly
    print("Processing fruit_jelly")
    copy_images_with_structure(
        src_dir='./mvtec_ad_2/fruit_jelly',
        dst_dir='./mvtec_ad_2_aug/fruit_jelly'
    )

    # rice
    print("Processing rice")
    root_dir = './mvtec_ad_2/rice'
    save_dir = './mvtec_ad_2_aug/rice'

    # train: 分别输出到 save_dir/train_long_edge 和 save_dir/train_grid
    train_dir = os.path.join(root_dir, 'train')

    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='grid',
        grid_size=(2,2),
        num_splits=2,
        save_format='png'
    )
    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='grid',
        grid_size=(4,4),
        num_splits=2,
        save_format='png'
    )
    # 其它子文件夹输出到 save_dir 下各自的_grid文件夹
    for sub in ['test_private', 'test_private_mixed', 'test_public', 'validation']:
        folder = os.path.join(root_dir, sub)
        grid_output = os.path.join(save_dir, sub)
        process_dataset(
            input_dir=folder,
            output_dir=grid_output,
            split_mode='grid',
            grid_size=(4,4),
            num_splits=2,
            save_format='png'
        )
    copy_images_with_structure(
        src_dir='./mvtec_ad_2/rice/train',
        dst_dir='./mvtec_ad_2_aug/rice/train'
    )

    # sheet_mental
    print("Processing sheet_mental")
    root_dir = './mvtec_ad_2/sheet_metal'
    save_dir = './mvtec_ad_2_aug/sheet_metal'

    # train: 分别输出到 save_dir/train_long_edge 和 save_dir/train_grid
    train_dir = os.path.join(root_dir, 'train')

    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='long_edge',
        grid_size=(2,2),
        num_splits=4,
        save_format='png'
    )
    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='grid',
        grid_size=(8,2),
        num_splits=2,
        save_format='png'
    )

    # 其它子文件夹输出到 save_dir 下各自的_grid文件夹
    for sub in ['test_private', 'test_private_mixed', 'test_public', 'validation']:
        folder = os.path.join(root_dir, sub)
        grid_output = os.path.join(save_dir, sub)
        process_dataset(
            input_dir=folder,
            output_dir=grid_output,
            split_mode='grid',
            grid_size=(8,2),
            num_splits=2,
            save_format='png'
        )

    copy_images_with_structure(
        src_dir='./mvtec_ad_2/sheet_metal/train',
        dst_dir='./mvtec_ad_2_aug/sheet_metal/train'
    )

    # vial
    print("Processing vial")
    copy_images_with_structure(
        src_dir='./mvtec_ad_2/vial',
        dst_dir='./mvtec_ad_2_aug/vial'
    )

    # wallplugs
    print("Processing wallplugs")
    copy_images_with_structure(
        src_dir='./mvtec_ad_2/wallplugs',
        dst_dir='./mvtec_ad_2_aug/wallplugs'
    )

    ##walnuts
    print("Processing walnuts")
    root_dir = './mvtec_ad_2/walnuts'
    save_dir = './mvtec_ad_2_aug/walnuts'

    # train: 分别输出到 save_dir/train_long_edge 和 save_dir/train_grid
    train_dir = os.path.join(root_dir, 'train')

    process_dataset(
        input_dir=train_dir,
        output_dir=os.path.join(save_dir, 'train'),
        split_mode='grid',
        grid_size=(2,2),
        num_splits=2,
        save_format='png'
    )

    # 其它子文件夹输出到 save_dir 下各自的_grid文件夹
    for sub in ['test_private', 'test_private_mixed', 'test_public', 'validation']:
        folder = os.path.join(root_dir, sub)
        grid_output = os.path.join(save_dir, sub)
        process_dataset(
            input_dir=folder,
            output_dir=grid_output,
            split_mode='grid',
            grid_size=(2,2),
            num_splits=2,
            save_format='png'
        )

    copy_images_with_structure(
        src_dir='./mvtec_ad_2/walnuts/train',
        dst_dir='./mvtec_ad_2_aug/walnuts/train'
    )