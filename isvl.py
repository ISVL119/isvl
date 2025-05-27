import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
import random
from torchvision import transforms
from optimizers import StableAdamW
from utils import evaluation_batch,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

# Dataset-Related Modules
from mvtec2_dataset import MVTec2Dataset
from mvtec2_dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from utils import get_gaussian_kernel, cal_anomaly_maps

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# 图像处理相关
from PIL import Image, ImageEnhance
import cv2

warnings.filterwarnings("ignore")


class RandomLightingAugmentation:
    def __init__(self, 
                 region_num=3, 
                 region_size_range=(30, 70), 
                 brightness_range=(0.5, 1.5),
                 left_right_brightness=(1.5, 0.7),
                 color_jitter_strength=(0.5, 1.2),
                 p=1.0,
                 restricted_classes=None):  # 新增：限制的类别
        self.region_num = region_num
        self.region_size_range = region_size_range
        self.brightness_range = brightness_range
        self.left_brightness = left_right_brightness[0]
        self.right_brightness = left_right_brightness[1]
        self.color_jitter = transforms.ColorJitter(brightness=color_jitter_strength)
        self.p = p
        self.restricted_classes = restricted_classes if restricted_classes else []

    def __call__(self, image, target_class=None):
        if random.random() > self.p:
            return image

        if target_class in self.restricted_classes:
            method = 'jitter'
        else:
            method = random.choice(['jitter', 'multi_region', 'left_right'])

        if method == 'jitter':
            return self.color_jitter(image)
        elif method == 'multi_region':
            return self.adjust_brightness_in_multiple_regions(image)
        elif method == 'left_right':
            return self.enhance_left_bright_right_dark(image)
        else:
            return image

    def adjust_brightness_in_multiple_regions(self, image):
        img_w, img_h = image.size
        new_image = image.copy()
        for i in range(self.region_num):
            region_w = random.randint(*self.region_size_range)
            region_h = random.randint(*self.region_size_range)
            left = random.randint(0, img_w - region_w)
            upper = random.randint(0, img_h - region_h)
            right = left + region_w
            lower = upper + region_h
            region = (left, upper, right, lower)
            brightness_factor = random.uniform(*self.brightness_range)
            cropped = new_image.crop(region)
            enhancer = ImageEnhance.Brightness(cropped)
            enhanced_crop = enhancer.enhance(brightness_factor)
            new_image.paste(enhanced_crop, region)
        return new_image

    def enhance_left_bright_right_dark(self, image):
        width, height = image.size
        mid = width // 2
        left_img = image.crop((0, 0, mid, height))
        right_img = image.crop((mid, 0, width, height))
        left_enhanced = ImageEnhance.Brightness(left_img).enhance(self.left_brightness)
        right_enhanced = ImageEnhance.Brightness(right_img).enhance(self.right_brightness)
        new_image = Image.new("RGB", (width, height))
        new_image.paste(left_enhanced, (0, 0))
        new_image.paste(right_enhanced, (mid, 0))
        return new_image

def find_best_threshold(anomaly_maps, gt_masks, step=1, verbose=True):
    """
    输入:
        anomaly_maps: List of np.array, 每张 (H,W)，范围是 [0,255] 的 anomaly_map_gray
        gt_masks: List of np.array, 每张 (H,W)，二值 0/255 的ground truth mask
        step: int, optional, 阈值搜索步长，默认为1
        verbose: bool, optional, 是否显示进度条
    输出:
        best_threshold: int, 0-255之间，使F1 score最高的阈值
        best_f1: float, 对应的最高F1分数
    """
    if not anomaly_maps or not gt_masks:
        raise ValueError("anomaly_maps 和 gt_masks 不能为空！")
    
    # Flatten and concatenate all predictions and gts
    all_preds = np.concatenate([x.flatten() for x in anomaly_maps]).astype(np.uint8)
    all_gts = np.concatenate([x.flatten() for x in gt_masks]).astype(np.uint8)
    all_gts = (all_gts > 127).astype(np.uint8)  # Normalize ground truth to {0,1}

    thresholds = np.arange(0, 256, step)
    best_f1 = 0
    best_threshold = 0

    # 预先算好正样本数量
    total_positives = np.sum(all_gts)

    if verbose:
        thresholds = tqdm(thresholds, desc="Searching best threshold")

    for threshold in thresholds:
        preds = (all_preds >= threshold).astype(np.uint8)

        tp = np.sum(preds * all_gts)
        fp = np.sum(preds) - tp
        fn = total_positives - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    data_train_transform_2 = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        RandomLightingAugmentation(p=0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_train_transform =  transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.5, 1.2))], p=0.5), 
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data_list = []
    test_data_list = []
    val_data_list = []
    true_val_data_list = []
    for i, item in enumerate(args.item_list):
        train_path = os.path.join(args.data_path, item)
        test_path = os.path.join(args.data_path, item)
        val_path = os.path.join(args.data_path, item)

        if item in ['can','fabric','rice']:
            train_data = MVTec2Dataset(root=train_path, transform=data_train_transform_2, gt_transform=gt_transform, phase="train",resize=args.input_size, normal_only=True)
        elif item in ['vial']:
            train_data = MVTec2Dataset(root=train_path, transform=data_train_transform, gt_transform=gt_transform, phase="train",resize=args.input_size, normal_only=True)
        else:
            train_data = MVTec2Dataset(root=train_path, transform=data_transform, gt_transform=gt_transform, phase="train",resize=args.input_size, normal_only=True)
        test_data = MVTec2Dataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test",resize=args.input_size, normal_only=True)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        val_data = MVTec2Dataset(root=val_path, transform=data_transform, gt_transform=gt_transform, phase="val",resize=args.input_size, normal_only=True)
        val_data_list.append(val_data)
        true_val_data = MVTec2Dataset(root=val_path, transform=data_transform, gt_transform=gt_transform, phase="true_val",resize=args.input_size, normal_only=True)
        true_val_data_list.append(true_val_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
    # fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, gt, label, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss,agg_prototype = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))
            if (epoch + 1) % args.total_epochs == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(args.item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, save_dir='./save_dir',dataset_root=args.data_path)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, anomaly_map_list, gt_mask_list = results
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
                model.train()
    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        model.eval()
        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, save_dir='./save_dir', dataset_root=args.data_path,normalize_amap=False)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, anomaly_map_list, gt_mask_list = results
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
            best_threshold, best_f1 = find_best_threshold(anomaly_map_list, gt_mask_list)
            print(f"Best threshold for {item} is {best_threshold} with F1 score {best_f1:.4f}")

        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
        
    elif args.phase == 'val':
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        model.eval()
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        save_dir = './results/anomaly_images'
        os.makedirs(save_dir, exist_ok=True)  # 创建根目录

        for item, val_data in zip(args.item_list, val_data_list):
            val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            with torch.no_grad():
                for img, gt, label, img_path_batch in tqdm(val_dataloader, ncols=80):
                    img = img.to(device)
                    output = model(img)
                    en, de = output[0], output[1]
                    original_shapes = []
                    for img_path in img_path_batch:
                        with Image.open(img_path) as im:
                            w, h = im.size
                            original_shapes.append((h, w))  # 注意换成 (height, width)
                    
                    out_size = original_shapes[0]

                    anomaly_map_batch, _ = cal_anomaly_maps(en, de, out_size)

                    anomaly_map_batch = gaussian_kernel(anomaly_map_batch)

                    anomaly_scores = []
                    for idx in range(anomaly_map_batch.shape[0]):
                        anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                        anomaly_map = np.clip(anomaly_map, 0, 1)
                        anomaly_scores.append(anomaly_map)
                    anomaly_scores = np.array(anomaly_scores)

                    for idx, img_path in enumerate(img_path_batch):
                        anomaly_map = anomaly_scores[idx]

                        relative_path = os.path.relpath(img_path, start=args.data_path)
                        save_path = os.path.join(save_dir, relative_path)
                        save_path = save_path.replace('.png', '')
                        save_dirname = os.path.dirname(save_path)
                        os.makedirs(save_dirname, exist_ok=True)

                        anomaly_map_gray = (anomaly_map * 255).astype(np.uint8)
                        cv2.imwrite(save_path + '.tiff', anomaly_map_gray)

    elif args.phase == 'true_val':
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        model.eval()
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

        all_anomaly_scores = []

        for item, true_val_data in zip(args.item_list, true_val_data_list):
            true_val_dataloader = torch.utils.data.DataLoader(true_val_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            with torch.no_grad():
                for img, gt, label, img_path_batch in tqdm(true_val_dataloader, ncols=80):
                    img = img.to(device)
                    output = model(img)
                    en, de = output[0], output[1]
                    original_shapes = []
                    for img_path in img_path_batch:
                        with Image.open(img_path) as im:
                            w, h = im.size
                            original_shapes.append((h, w))  # 注意换成 (height, width)
                    
                    out_size = original_shapes[0]

                    anomaly_map_batch, _ = cal_anomaly_maps(en, de, out_size)

                    anomaly_map_batch = gaussian_kernel(anomaly_map_batch)

                    for idx in range(anomaly_map_batch.shape[0]):
                        anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                        anomaly_map = np.clip(anomaly_map, 0, 1)
                        all_anomaly_scores.append(anomaly_map)

        all_anomaly_scores_np = np.stack(all_anomaly_scores, axis=0)  # 形状: [N, H, W]

        mean_value = np.mean(all_anomaly_scores_np) * 255
        std_value = np.std(all_anomaly_scores_np) * 255

        print("所有anomaly map像素的均值: ", mean_value)
        print("所有anomaly map像素的方差: ", std_value)



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'mvtec_ad_2') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'./mvtec_ad_2_aug') # Replace it with your path. E:\Dataset\mvtec_ad_2_aug

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_large_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    # category_info
    parser.add_argument('--item_list', nargs='+', default=['can'], help='item列表（空格分隔）')

    args = parser.parse_args()

    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}_CLASS={args.item_list[0]}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    main(args)
