# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Set

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Define the date format
)

logger = logging.getLogger()

OBJECT_FILE_COUNTER = {
    'can': 321,
    'fabric': 314,
    'fruit_jelly': 255,
    'rice': 277,
    'sheet_metal': 142,
    'vial': 276,
    'wallplugs': 232,
    'walnuts': 228,
}

MVTEC_AD_2_OBJECTS = set(OBJECT_FILE_COUNTER.keys())

TEST_SET_DIRECTORIES = {'test_private', 'test_private_mixed'}

DIRECTORY_STRUCTURE = """
submission_folder/
    |-- anomaly_images/
        |-- can/
            |-- test_private/
                |-- 000_regular.tiff
                | ...
                |-- 320_regular.tiff
            |-- test_private_mixed/
                |-- 000_mixed.tiff
                | ...
                |-- 320_mixed.tiff
        |-- fabric
            |-- test_private/
                |-- 000_regular.tiff
                | ...
            |-- test_private_mixed/
                |-- 000_mixed.tiff
                | ...
        |-- fruit_jelly
        |-- rice
        |-- sheet_metal
        |-- vial
        |-- wallplugs
        |-- walnuts
    |-- anomaly_images_thresholded/
        |-- can/
            |-- test_private/
                |-- 000_regular.png
                | ...
            |-- test_private_mixed/
                |-- 000_mixed.png
                | ...
        |-- fabric
        |-- fruit_jelly
        |-- rice
        |-- sheet_metal
        |-- vial
        |-- wallplugs
        |-- walnuts
"""


class SubmissionException(Exception):

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def compare_found_vs_required(
    required_dirs: set, required_files: set, parent_dir: Path
) -> None:
    """
    Compares found directories/files against required ones.

    Args:
        required_dirs (set): A set of expected directory names.
        required_files (set): A set of required file names.
        parent_dir (Path): The parent directory being checked.

    Raises:
        SubmissionException: For missing/spurious directory content.
    """
    found_files, found_dirs = set(), set()

    for item in parent_dir.iterdir():
        if item.is_file():
            found_files.add(item.name)
        elif item.is_dir():
            found_dirs.add(item.name)

    missing_dirs = required_dirs - found_dirs
    redundant_dirs = found_dirs - required_dirs
    missing_files = required_files - found_files
    redundant_files = found_files - required_files

    if redundant_dirs:
        raise SubmissionException(
            f"Superfluous directories: "
            f"{_print_mismatched_files(redundant_dirs)} in "
            f"{parent_dir.as_posix()}.\nPlease adhere to the directory "
            f"structure:\n{DIRECTORY_STRUCTURE}"
        )

    if missing_dirs:
        raise SubmissionException(
            f"Missing directories: {_print_mismatched_files(missing_dirs)} "
            f"in {parent_dir.as_posix()}."
        )

    if redundant_files:
        raise SubmissionException(
            f"Superfluous files: {_print_mismatched_files(redundant_files)} "
            f"in {parent_dir.as_posix()}.\nPlease adhere to the directory "
            f"structure:\n{DIRECTORY_STRUCTURE}"
        )

    if missing_files:
        raise SubmissionException(
            f"Missing files: {_print_mismatched_files(missing_files)} in "
            f"{parent_dir.as_posix()}."
        )


def _print_mismatched_files(mismatched_files: Set[str]) -> str:
    """Nicely formats the list of mismatched files, limiting the number
    displayed.

    Args:
        mismatched_files (Set[str]): Set of mismatched file names.

    Returns:
        str: A formatted string.
    """
    num_files_to_print = 4
    mismatched_files = sorted(mismatched_files)
    num_mismatched_files = len(mismatched_files)
    if num_mismatched_files <= num_files_to_print:
        return f"[{', '.join(mismatched_files)}]"
    return (
        f"[{', '.join(mismatched_files[:num_files_to_print])}, ...] + "
        f"{num_mismatched_files - num_files_to_print} more"
    )


def _check_anomaly_images(
    mad2_object: str,
    parent_dir: Path,
    expected_file_format: str,
) -> List[Path]:
    """
    Checks if the image files in the directory follow the required structure.

    Args:
        mad2_object (str): The dataset object name.
        parent_dir (Path): The directory containing the image files.
        expected_file_format (str): Expected file format of image files.

    Raises:
        SubmissionException: If image files do not follow exepected structure.

    Returns:
        List[Path]: List of found image file paths.
    """
    suffix = "regular" if parent_dir.name == "test_private" else "mixed"
    files = list(parent_dir.iterdir())
    num_expected_files = OBJECT_FILE_COUNTER[mad2_object]
    if len(files) != num_expected_files:
        raise SubmissionException(
            (
                f"Expected {num_expected_files} files, found "
                f"{len(files)} in {parent_dir.as_posix()}."
            )
        )
    if not all(file.suffix == expected_file_format for file in files):
        raise SubmissionException(
            (
                f"Expected only {expected_file_format} files in "
                f"{parent_dir.as_posix()}."
            )
        )

    expected_files = {
        f"{str(idx).zfill(3)}_{suffix}{expected_file_format}"
        for idx in range(num_expected_files)
    }
    compare_found_vs_required(
        set(),
        expected_files,
        parent_dir,
    )

    return files


def check_anomaly_image_dir(
    anomaly_image_dir_path: Path,
    expected_file_format: str,
) -> List[Path]:
    """Checks the structure and content of the anomaly image directory.

    Args:
        anomaly_image_dir_path (Path): Path to the anomaly images directory.
        expected_file_format (str): Expected file format of anomaly images
        (.png | .tiff)

    Returns:
        List[Path]: List of Paths to anomaly images.
    """
    anomaly_image_paths = []
    compare_found_vs_required(
        MVTEC_AD_2_OBJECTS, set(), anomaly_image_dir_path
    )

    # Check if each object directory contains the respective test sets
    for mad2_object in sorted(MVTEC_AD_2_OBJECTS):
        mad2_object_dir = anomaly_image_dir_path / mad2_object
        compare_found_vs_required(TEST_SET_DIRECTORIES, set(), mad2_object_dir)

        for test_folder_name in TEST_SET_DIRECTORIES:
            test_folder = mad2_object_dir / test_folder_name
            found_paths = _check_anomaly_images(
                mad2_object,
                test_folder,
                expected_file_format,
            )
            anomaly_image_paths.extend(found_paths)
    return anomaly_image_paths


def check_images(image_paths: List[Path], thresholded: bool) -> None:
    """Verifies format and content of the images.

    Args:
        image_paths (List[Path]): List of file paths of (thresholded) anomaly
          images.
        thresholded (bool): Whether images are thresholded.
    """
    checker_fn = (
        _check_thresholded_ad_images if thresholded else _check_ad_images
    )
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(checker_fn, img_path) for img_path in image_paths
        ]
        for future in futures:
            future.result()


def _check_ad_images(
    anomaly_image_path: Path, expected_data_type=np.float16
) -> None:
    """Verifies that anomaly image is single-channel image of expected data
    format.

    Args:
        anomaly_image_path (Path): File path of anomaly image.
        expected_data_type (np.dtype): Expected data type of anomaly images

    Raises:
        SubmissionException: If image is multi-channel image or has an
          unexpected data format.
    """
    img = tifffile.imread(anomaly_image_path)

    if not img.ndim == 2:
        raise SubmissionException(
            f"Anomaly image {anomaly_image_path.as_posix()} ist not a "
            f"single-channel image."
        )

    if img.dtype != expected_data_type:
        raise SubmissionException(
            f"Anomaly image {anomaly_image_path.as_posix()} is not of type "
            f"{expected_data_type.__name__}. Please convert the images in "
            f"directory anomaly_images before uploading."
        )


def _check_thresholded_ad_images(thresh_anomaly_image_path: Path) -> None:
    """Verifies that thresholded image is single-channel image with pixel
    values in {0, 255}.

    Args:
        thresh_anomaly_image_path (Path): File path of thresholded anomaly
          image.

    Raises:
        SubmissionException: If image is non-binary or multi-channel image.
    """
    img = np.asarray(Image.open(thresh_anomaly_image_path))

    if not img.ndim == 2:
        raise SubmissionException(
            f"Thresholded image {thresh_anomaly_image_path.as_posix()} is not "
            f"a single-channel image."
        )

    unique_values = np.unique(img)
    if not np.all(np.isin(unique_values, [0, 255])):
        raise SubmissionException(
            f"Values of thresholded image "
            f"{thresh_anomaly_image_path.as_posix()} are not in {{0, 255}}. "
            f"Please ensure that anomaly-free pixels are set to zero and "
            f"anomalous ones to 255."
        )


def compress_submission(submission_file_path: str):
    """Compresses the given submission.

    Args:
        submission_file_path (str): Path to the submission.
    """
    logger.info(f"Creating compressed version of {submission_file_path}.")
    output_name = Path(submission_file_path).name
    output_file = f'./{output_name}.tar.gz'

    # Create a list of files to be archived
    files_to_compress = []
    for root, _, files in os.walk(submission_file_path):
        for file in files:
            files_to_compress.append(Path(root, file))
    total_files = len(files_to_compress)

    with tarfile.open(output_file, "w:gz") as tar:
        for file in tqdm(
            files_to_compress,
            desc="Compressing files",
            unit="files",
            total=total_files,
        ):
            tar.add(
                file,
                arcname=Path(
                    output_name,
                    file.relative_to(submission_file_path),
                ),
            )
    logger.info(
        f"Successfully compressed the given directory to {output_file}"
    )
