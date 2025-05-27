"""
Script to measure inference runtime [seconds] and GPU memory consumption [MiB].
Note that this script does not support measuring memory consumption when 
deploying models on CPU.

general usage:
- specify OUT_DIR
- insert your model and forward pass call (see # INSERT)
- run this script
"""

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

USE_GPU = True
BATCH_SIZE = 1
WARMUP_ITERATIONS_GPU = 1000
TIMING_ITERATIONS_GPU = 1000
TIMING_ITERATIONS_CPU = 1000  # no warmup for CPU
OUT_DIR = '/info/is/saved/here'  # INSERT


class InfiniteDataset(torch.utils.data.IterableDataset):

    def __init__(self, image_height=256, image_width=256, dtype=torch.float32):
        self.dtype = dtype
        self.image_height = image_height
        self.image_width = image_width
        super().__init__()

    def __iter__(self):
        while True:
            image_np = np.random.randn(3, self.image_height, self.image_width)
            image_pt = torch.as_tensor(image_np, dtype=self.dtype)
            yield image_pt


def main():

    device = torch.device('cpu')
    device_name = 'cpu'
    if USE_GPU:
        assert torch.cuda.is_available(), 'GPU not available.'
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        device_name = 'gpu'

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Using device {device}')

    torch.set_grad_enabled(False)

    measurement_info = {
        'image_height': [],
        'image_width': [],
        'runtime_mean': [],
        'runtime_std': [],
        'runtime_min': [],
        'runtime_max': [],
        'peak_memory': [],
    }

    num_iterations = (
        WARMUP_ITERATIONS_GPU + TIMING_ITERATIONS_GPU
        if USE_GPU
        else TIMING_ITERATIONS_CPU
    )

    for image_size in [(256, 256), (512, 512), (1024, 1224)]:

        img_height, img_width = image_size
        measurement_info['image_height'].append(img_height)
        measurement_info['image_width'].append(img_width)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # INSERT
        # create your model instance here
        model = 'MODEL_CLASS'
        model.to(device).eval()

        dataset = InfiniteDataset(
            image_height=img_height,
            image_width=img_width,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True
        )

        dataiter = iter(dataloader)
        times = []

        timing_successful = True
        print(f'\nInference on image size (hxw): {img_height}x{img_width}')
        try:
            for _ in tqdm(range(num_iterations)):
                input_tensor = next(dataiter)

                # START
                start_time = time.time()

                input_tensor = input_tensor.to(device)

                # INSERT
                # call your forward pass and ensure
                # that in the end you have an anomaly image on the cpu
                anomaly_image = model(input_tensor)
                anomaly_image = anomaly_image.cpu()

                # END
                # we stop the timing as soon as we have the result on the cpu
                end_time = time.time()
                times.append((end_time - start_time) / BATCH_SIZE)
        except Exception as error:
            timing_successful = False
            print('timing not successful:', error)

            error_value = np.nan
            measurement_info['runtime_mean'].append(error_value)
            measurement_info['runtime_std'].append(error_value)
            measurement_info['runtime_min'].append(error_value)
            measurement_info['runtime_max'].append(error_value)
            measurement_info['peak_memory'].append(error_value)

        if timing_successful:

            if USE_GPU:
                # get peak memory
                peak_memory = torch.cuda.memory_stats()[
                    'reserved_bytes.all.peak'
                ] / (1024**2)
                measurement_info['peak_memory'].append(peak_memory)
                print('Peak Memory:', peak_memory)

                # discard warmup iterations
                times = times[WARMUP_ITERATIONS_GPU:]

            else:
                measurement_info['peak_memory'].append(np.nan)

            print('Mean runtime [s]:', np.mean(times))
            print('Std:', np.std(times))
            print('Min:', np.min(times))
            print('Max:', np.max(times))
            print('Last:', times[-1])

            measurement_info['runtime_mean'].append(np.mean(times))
            measurement_info['runtime_std'].append(np.std(times))
            measurement_info['runtime_min'].append(np.min(times))
            measurement_info['runtime_max'].append(np.max(times))

    # save the data as csv
    measurement_data = pd.DataFrame.from_dict(
        measurement_info, orient='columns'
    )
    measurement_data.to_csv(
        os.path.join(OUT_DIR, f'runtimes_and_memory_{device_name}.csv'),
        sep=',',
        index=False,
    )


if __name__ == '__main__':
    main()
