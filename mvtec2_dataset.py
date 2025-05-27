import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import re

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTec2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase,resize, normal_only=True):
        self.phase = phase
        self.resize = resize
        self.transform = transform
        self.root = root
        self.gt_transform = gt_transform
        self.normal_only = normal_only

        if phase == 'train':
            # self.img_path = os.path.join(root, 'train', 'good')
            # self.gt_path = None
            self.img_path_good = os.path.join(root, 'train', 'good')
            self.img_path_bad = os.path.join(root, 'train', 'bad')
            self.gt_path_bad = os.path.join(root, 'train', 'bad_mask')

        elif phase == 'test':
            self.img_path_good = os.path.join(root, 'test_public', 'good')
            self.img_path_bad = os.path.join(root, 'test_public', 'bad')
            self.gt_path_bad = os.path.join(root, 'test_public', 'ground_truth', 'bad')

        elif phase == 'val':
            self.img_path_private = os.path.join(root, 'test_private')
            self.img_path_private_mixed = os.path.join(root, 'test_private_mixed')
            self.gt_path = None

        elif phase == 'true_val':
            self.img_path_good = os.path.join(root, 'validation', 'good')

        else:
            raise ValueError(f"Unsupported phase: {phase}")

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def generate_gt_filename(self,filename):
        name, ext = os.path.splitext(filename)
        # 如果已经有 _mask，直接返回原名
        if '_mask' in name:
            return name + ext

        # 先找分割的后缀
        # 支持 _grid4x2_0_0, _longedge4_3 等多种新后缀
        pattern = r'(_grid\d+x\d+_\d+_\d+|_longedge\d+_\d+)$'
        match = re.search(pattern, name)
        if match:
            idx = match.start()
            prefix = name[:idx]
            suffix = name[idx:]
            gt_name = prefix + '_mask' + suffix
        else:
            gt_name = name + '_mask'

        return gt_name + ext

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if self.phase == 'train':
            # 加载 good 图像
            good_paths = glob.glob(os.path.join(self.img_path_good, '*.png')) + \
                        glob.glob(os.path.join(self.img_path_good, '*.jpg')) + \
                        glob.glob(os.path.join(self.img_path_good, '*.bmp'))
            
            img_tot_paths.extend(good_paths)
            gt_tot_paths.extend([0] * len(good_paths))  # 没有 GT，为 0
            tot_labels.extend([0] * len(good_paths))
            tot_types.extend(['good'] * len(good_paths))

            if not self.normal_only:
                # 加载 bad 图像
                bad_paths = glob.glob(os.path.join(self.img_path_bad, '*.png')) + \
                            glob.glob(os.path.join(self.img_path_bad, '*.jpg')) + \
                            glob.glob(os.path.join(self.img_path_bad, '*.bmp'))
                bad_paths.sort()

                for img_path in bad_paths:
                    filename = os.path.basename(img_path)
                    gt_filename = self.generate_gt_filename(filename)
                    gt_path = os.path.join(self.gt_path_bad, gt_filename)

                    if not os.path.exists(gt_path):
                        raise FileNotFoundError(f"Missing GT mask: {gt_path}")
                    
                    mask = np.array(Image.open(gt_path).convert('L'))

                    if not np.any(mask):
                        label = 0
                        type_str = 'good'
                    else:
                        label = 1
                        type_str = 'bad'

                    img_tot_paths.append(img_path)
                    gt_tot_paths.append(gt_path)
                    tot_labels.append(label)
                    tot_types.append(type_str)

        elif self.phase == 'test':
            # Load good test images
            good_paths = glob.glob(os.path.join(self.img_path_good, '*.png')) + \
                         glob.glob(os.path.join(self.img_path_good, '*.jpg')) + \
                         glob.glob(os.path.join(self.img_path_good, '*.bmp'))
            img_tot_paths.extend(good_paths)
            gt_tot_paths.extend([0] * len(good_paths))
            tot_labels.extend([0] * len(good_paths))
            tot_types.extend(['good'] * len(good_paths))

            # Load bad test images
            bad_paths = glob.glob(os.path.join(self.img_path_bad, '*.png')) + \
                        glob.glob(os.path.join(self.img_path_bad, '*.jpg')) + \
                        glob.glob(os.path.join(self.img_path_bad, '*.bmp'))
            bad_paths.sort()

            for img_path in bad_paths:
                filename = os.path.basename(img_path)
                gt_filename = self.generate_gt_filename(filename)
                gt_path = os.path.join(self.gt_path_bad, gt_filename)

                if not os.path.exists(gt_path):
                    raise FileNotFoundError(f"Ground truth mask not found: {gt_path}")
                
                mask = np.array(Image.open(gt_path).convert('L'))
                
                if not np.any(mask):
                    label = 0
                    type_str = 'good'
                else:
                    label = 1
                    type_str = 'bad'

                img_tot_paths.append(img_path)
                gt_tot_paths.append(gt_path)
                tot_labels.append(label)
                tot_types.append(type_str)
        
        elif self.phase == 'val':
            private_paths = glob.glob(os.path.join(self.img_path_private, '*.png')) + \
                            glob.glob(os.path.join(self.img_path_private, '*.jpg')) + \
                            glob.glob(os.path.join(self.img_path_private, '*.bmp'))
            private_mixed_paths = glob.glob(os.path.join(self.img_path_private_mixed, '*.png')) + \
                                  glob.glob(os.path.join(self.img_path_private_mixed, '*.jpg')) + \
                                  glob.glob(os.path.join(self.img_path_private_mixed, '*.bmp'))
            img_paths = private_paths + private_mixed_paths
            img_paths.sort()

            img_tot_paths.extend(img_paths)
            gt_tot_paths.extend([0] * len(img_paths))
            tot_labels.extend([0] * len(img_paths))
            tot_types.extend(['good'] * len(img_paths))
        
        elif self.phase == 'true_val':
            good_paths = glob.glob(os.path.join(self.img_path_good, '*.png')) + \
            glob.glob(os.path.join(self.img_path_good, '*.jpg')) + \
            glob.glob(os.path.join(self.img_path_good, '*.bmp'))
            
            img_tot_paths.extend(good_paths)
            gt_tot_paths.extend([0] * len(good_paths))  # 没有 GT，为 0
            tot_labels.extend([0] * len(good_paths))
            tot_types.extend(['good'] * len(good_paths))
        
        else:
            raise ValueError(f"Unsupported phase: {self.phase}")

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-1]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], f"Image and GT size mismatch: {img.size()} vs {gt.size()}"

        return img, gt, label, img_path

