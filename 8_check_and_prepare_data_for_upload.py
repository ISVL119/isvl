"""
This script verifies your submission for completeness and checks for common
formatting errors. Run this script before uploading your submission to the
evaluation server to avoid reductions in your evaluation budget due to minor
mistakes. When all checks were successful, the submission directory is
compressed.
"""

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
from pathlib import Path

from submit_utils import (
    DIRECTORY_STRUCTURE,
    SubmissionException,
    check_anomaly_image_dir,
    check_images,
    compare_found_vs_required,
    compress_submission,
    logger,
)


def check_submission(submission_file_path: str) -> None:
    """
    Checks the structure and content of the submission directory.

    Args:
        submission_file_path (str): Path to the submission directory.

    Raises:
        SubmissionException: If directory structure or content is incorrect.
    """
    logger.info(f"Start checking submission {submission_file_path}")
    submission_file_path = Path(submission_file_path)
    if not Path(submission_file_path).is_dir():
        raise SubmissionException("The given path is not a directory")

    required_ad_image_dirs = {'anomaly_images'}

    root_ad_images = submission_file_path / 'anomaly_images'
    if not root_ad_images.exists():
        raise SubmissionException(
            f"{root_ad_images.as_posix()} was not found. Please adhere to the "
            f"directory structure: \n{DIRECTORY_STRUCTURE}"
        )

    logger.info("Check structure and content of the anomaly_image directory")
    ad_image_paths = check_anomaly_image_dir(
        root_ad_images,
        expected_file_format='.tiff',
    )
    check_images(ad_image_paths, thresholded=False)
    logger.info("Done")

    root_thresh_ad_images = submission_file_path / 'anomaly_images_thresholded'
    if not root_thresh_ad_images.exists():
        logger.warning(
            f"{root_thresh_ad_images.as_posix()} was not found. To "
            f"evaluate threshold-dependent metrics, binarize the anomaly "
            f"images: Set normal pixels to 0 and anomalous pixels to 255. You "
            f"could start with this baseline method: segmentation_threshold = "
            f"np.mean(anomaly_scores_val) + 3 * np.std(anomaly_scores_val)"
        )
    else:
        logger.info(
            "Check structure and content of the anomaly_images_thresholded "
            "directory"
        )
        thresh_ad_image_paths = check_anomaly_image_dir(
            root_thresh_ad_images, expected_file_format='.png'
        )
        check_images(thresh_ad_image_paths, thresholded=True)
        logger.info("Done")

        required_ad_image_dirs.add('anomaly_images_thresholded')

    compare_found_vs_required(
        required_ad_image_dirs, set(), submission_file_path
    )
    logger.info("All checks successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "submission_path",
        type=str,
        help=(
            "Path to your submission directory containing all MVTec AD 2 "
            "objects."
        ),
    )
    args = parser.parse_args()

    check_submission(args.submission_path)

    compress_submission(args.submission_path)
