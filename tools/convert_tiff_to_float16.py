import numpy as np
from PIL import Image
import tifffile
from pathlib import Path
from tqdm import tqdm

def convert_images_to_float16_tiff_and_cleanup(root_dir: str):
    """
    Converts all .tiff, .png, .jpg, .jpeg files under root_dir to float16 .tiff format.
    After successful conversion, deletes the original file if it was not already a .tiff/.tif.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Provided directory does not exist: {root_dir}")

    valid_extensions = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
    image_files = [p for p in root_path.rglob("*") if p.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"No image files found under {root_dir}.")
        return

    for img_path in tqdm(image_files, desc="Converting images to float16 TIFF"):
        try:
            # 1. 读取图像
            suffix = img_path.suffix.lower()
            if suffix in [".tiff", ".tif"]:
                img = tifffile.imread(img_path)
            else:
                img = np.array(Image.open(img_path))

            # 2. 转 dtype
            if img.dtype != np.float16:
                img = img.astype(np.float16)

            # 3. 生成输出路径
            if suffix in [".tiff", ".tif"]:
                output_path = img_path
            else:
                output_path = img_path.with_suffix(".tiff")

            # 4. 写出 TIFF
            tifffile.imwrite(output_path, img, dtype=np.float16)

            # 5. 如果原文件不是 TIFF，就删除它
            if suffix not in [".tiff", ".tif"]:
                try:
                    img_path.unlink()
                except Exception as e_del:
                    print(f"[WARN] Failed to delete original file {img_path}: {e_del}")

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

    print(f"Conversion complete! Total {len(image_files)} files checked.")

def convert_images_to_float16_tiff_only(root_dir: str,
                                          output_dir_name: str = 'anomaly_images'):
    """
    Converts all .png, .jpg, .jpeg files under root_dir to float16 .tiff format,
    preserving the original images. The converted TIFF files are saved under a sibling
    directory named output_dir_name with the same subdirectory structure.

    Args:
        root_dir: Directory containing original images (e.g., anomaly_images_png).
        output_dir_name: Name of the directory (sibling to root_dir) to place .tiff files.
    """
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import tifffile
    from tqdm import tqdm

    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Provided directory does not exist: {root_dir}")

    # Define extensions to convert
    valid_extensions = ['.png', '.jpg', '.jpeg']
    # Collect files
    image_files = [p for p in root_path.rglob('*') if p.suffix.lower() in valid_extensions]
    if not image_files:
        print(f"No PNG/JPG/JPEG files found under {root_dir}.")
        return

    # Prepare output base directory: sibling of root_dir
    output_base = root_path.parent / output_dir_name

    for img_path in tqdm(image_files, desc="Converting to float16 TIFF"):
        try:
            # 1. Read image as numpy array
            img = np.array(Image.open(img_path))
            # 2. Convert dtype to float16 if needed
            if img.dtype != np.float16:
                img = img.astype(np.float16)

            # 3. Compute relative path inside root_dir, then create output path
            rel = img_path.relative_to(root_path)
            tiff_rel = rel.with_suffix('.tiff')
            output_path = output_base / tiff_rel
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 4. Write TIFF
            tifffile.imwrite(str(output_path), img, dtype=np.float16)

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

    print(f"Conversion complete! Converted {len(image_files)} files to {output_dir_name}.")


if __name__ == "__main__":
    directory = "/home/hy/xsy_pan/250522vand/anomaly_images/vial"
    convert_images_to_float16_tiff_and_cleanup(directory)
