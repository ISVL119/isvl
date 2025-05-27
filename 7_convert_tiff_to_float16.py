import numpy as np
from PIL import Image
import tifffile
from pathlib import Path
from tqdm import tqdm

def convert_images_to_float16_tiff(root_dir: str):
    """
    Converts all .tiff, .png, .jpg, .jpeg files under root_dir to float16 .tiff format.

    Args:
        root_dir (str): Root directory containing image files to convert.
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
            if img_path.suffix.lower() in [".tiff", ".tif"]:
                img = tifffile.imread(img_path)
            else:
                img = Image.open(img_path)
                img = np.array(img)

            if img.dtype != np.float16:
                img = img.astype(np.float16)

            # 构造新的tiff文件路径
                if img_path.suffix.lower() in [".tiff", ".tif"]:
                    output_path = img_path
                else:
                    output_path = img_path.with_suffix(".tiff")

                tifffile.imwrite(output_path, img, dtype=np.float16)

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

    print(f"Conversion complete! Total {len(image_files)} files checked.")

# 使用方法
if __name__ == "__main__":
    directory = "./results/anomaly_images"
    convert_images_to_float16_tiff(directory)

