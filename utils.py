import torch
from numpy.random import normal
import  random
import shutil
import sys
import time
import inspect
import logging
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
import os
from functools import partial
import math
from tqdm import tqdm
from aug_funcs import rot_img, translation_img, hflip_img, grey_img, rot90_img
import torch.backends.cudnn as cudnn
from adeval import  EvalAccumulatorCuda


def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics = ['I-AUROC', 'I-AP', 'I-F1_max','P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)
        
    score_min = min(pr_sp)
    score_max = max(pr_sp)
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()
    
    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
    
    # for i in range(torch.tensor(pr_px).size(0)):
    #     accum.add_image(torch.tensor(pr_sp[i]), torch.tensor(gt_sp[i]))
    
    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def augmentation(img):
    img = img.unsqueeze(0)
    augment_img = img
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(img, angle)
        augment_img = torch.cat([augment_img, rotate_img], dim=0)
        # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(img, a, b)
        augment_img = torch.cat([augment_img, trans_img], dim=0)
        # hflip img
    flipped_img = hflip_img(img)
    augment_img = torch.cat([augment_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(img)
    augment_img = torch.cat([augment_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(img, angle)
        augment_img = torch.cat([augment_img, rotate90_img], dim=0)
    augment_img = (augment_img[torch.randperm(augment_img.size(0))])
    return augment_img

def modify_grad(x, inds, factor=0.):
    # print(inds.shape)
    inds = inds.expand_as(x)
    # print(x.shape)
    # print(inds.shape)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

def masked_cosine_loss(a, b, mask, factor=5.0):
    """
    mask: [B, 1, H, W]，数值范围为0~1，异常区域为1
    a, b: list of feature maps, each of shape [B, C, H, W]
    """
    cos = torch.nn.CosineSimilarity(dim=1)
    total_loss = 0.0

    for i in range(len(a)):
        a_feat = a[i].detach()  # detach encoder side
        b_feat = b[i]

        sim_map = cos(a_feat, b_feat)  # shape: [B, H, W]
        dist_map = 1.0 - sim_map       # dissimilarity

        # Broadcast mask to [B, H, W]
        mask_resized = F.interpolate(mask, size=dist_map.shape[1:], mode='nearest')
        weight = 1.0 + factor * mask_resized.squeeze(1)  # shape: [B, H, W]

        weighted_loss = dist_map * weight
        mean_loss = weighted_loss.mean()

        total_loss += mean_loss

        # 可选：注册反向传播 hook 强化异常区域
        def mask_grad_hook(grad, weight=weight):
            return grad * (1.0 + weight.unsqueeze(1))  # broadcast to [B, C, H, W]

        b_feat.register_hook(partial(mask_grad_hook, weight=weight))

    return total_loss / len(a)

def gather_loss(query, keys):
    distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
    distance, cluster_index = torch.min(distribution, dim=2)
    gather_loss = distance.mean()
    return gather_loss

def semi_supervised_guided_loss(en, de, mask, alpha=0.7, beta=0.3, y=3.0):
    cos = torch.nn.CosineSimilarity(dim=1)
    total_loss = 0.0

    for e, d in zip(en, de):
        # with torch.no_grad():
        sim_map = cos(e.detach(), d)
        dist_map = 1.0 - sim_map  # 重建差

        mask_resized = F.interpolate(mask, size=dist_map.shape[1:], mode='nearest')
        abnormal_mask = mask_resized.squeeze(1)       # 异常区域 = 1
        normal_mask = 1.0 - abnormal_mask              # 正常区域 = 1

        mean_dist = dist_map.mean()
        factor = (dist_map / (mean_dist + 1e-6)) ** y  # 避免除0
        if factor.ndim == 3:
            factor = factor.unsqueeze(1)  # [B, 1, H, W]

        # Hook 放大异常区域梯度（只在 backward 有效）
        d.register_hook(partial(modify_grad_v2, factor=factor))

        # 计算正常区域 loss（希望重建好）与异常区域 loss（希望重建差）
        normal_loss = (dist_map * normal_mask).mean()
        abnormal_loss = (dist_map * abnormal_mask).mean()

        total_loss += alpha * normal_loss + beta * (1.0 - abnormal_loss)

    return total_loss / len(en)

def inp_guided_hybrid_loss(en, de, mask, alpha=0.7, beta=0.3):
    cos = torch.nn.CosineSimilarity(dim=1)
    total_loss = 0.0

    for e, d in zip(en, de):
        sim_map = cos(e.detach(), d)
        dist_map = 1.0 - sim_map
        mask_resized = F.interpolate(mask, size=dist_map.shape[1:], mode='nearest')
        normal_mask = 1.0 - mask_resized.squeeze(1)

        normal_loss = (dist_map * normal_mask).mean()
        abnormal_loss = (dist_map * mask_resized.squeeze(1)).mean()

        # ✅ 修复点：鼓励异常区域重建差 → loss 最大化
        total_loss += alpha * normal_loss + beta * (1.0 - abnormal_loss) 

    return total_loss / len(en)

def contrastive_patch_inp_loss(patches, inps, mask_patch, temperature=0.07):
    B, N, C = patches.shape
    M = inps.shape[1]
    sim = F.cosine_similarity(patches.unsqueeze(2), inps.unsqueeze(1), dim=-1) / temperature  # [B, N, M]
    loss = 0.0
    for b in range(B):
        pos_mask = mask_patch[b] < 0.01
        neg_mask = mask_patch[b] >= 0.01
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue  # 跳过此 batch 项

        pos = sim[b][pos_mask].mean()
        neg = sim[b][neg_mask].mean()
        loss += F.relu(neg - pos + 0.5)
    return loss / B

def anomaly_proto_loss(proto_normal, proto_abnormal):
    # proto_normal, proto_abnormal: [B, M, C]
    proto_dist = 1 - F.cosine_similarity(proto_normal, proto_abnormal, dim=-1)  # [B, M]
    return proto_dist.mean()

def focal_bce_with_logits(inputs, targets, gamma=2.0, alpha=0.25, reduction='none'):
    """
    Stable focal BCE loss for logits (no sigmoid required on inputs).
    inputs: raw logits from model (no sigmoid) — shape [B, 1, H, W]
    targets: binary ground truth mask (0. or 1.) — shape [B, 1, H, W]
    """

    # Ensure inputs are in valid range
    inputs = torch.clamp(inputs, min=1e-6, max=1 - 1e-6)  # Avoid log(0) instability

    # Compute BCE with logits (no reduction)
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

    
    # Compute pt = p_t
    pt = targets * inputs + (1 - targets) * (1 - inputs)
    focal_weight = (1 - pt) ** gamma

    # Class balancing
    alpha_weight = targets * alpha + (1 - targets) * (1 - alpha)

    # Final loss
    loss = alpha_weight * focal_weight * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction

def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        factor = (point_dist/mean_dist)**(y)
        # factor = factor/torch.max(factor)
        # factor = torch.clip(factor, min=min_grad)
        # print(thresh)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        # a_map = torch.clamp(a_map, min=0.0, max=1.0)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N

def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def save_imag_ZS(imgs, anomaly_map, gt, prototype_map, save_root, img_path):
    batch_num = imgs.shape[0]
    for i in range(batch_num):
        img_path_list = img_path[i].split('\\')
        class_name, category, idx_name = img_path_list[-4], img_path_list[-2], img_path_list[-1]
        os.makedirs(os.path.join(save_root, class_name, category), exist_ok=True)
        input_frame = denormalize(imgs[i].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_0.png'), cv2_input)
        ano_map = anomaly_map[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_1.png'), ano_map, cmap='jet')
        gt_map = gt[i].squeeze(0).cpu().detach().numpy()
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_2.png'), gt_map, cmap='gray')
        distance = prototype_map[i].view((28, 28)).cpu().detach().numpy()
        distance = cv2.resize(distance, (392, 392), interpolation=cv2.INTER_AREA)
        plt.imsave(os.path.join(save_root, class_name, category, fr'{idx_name}_3.png'), distance, cmap='jet')
        plt.close()

def save_concat_image(defect, target, anomaly_map_gray, save_path):
    """
    defect: numpy (H,W,3) 原图 (BGR格式)
    target: numpy (H,W,3) 真值Mask (BGR格式)
    anomaly_map_gray: numpy (H,W) anomaly map灰度图(0-255)
    save_path: 保存路径 (不带后缀)
    """

    # anomaly map 灰度图转伪彩色
    anomaly_map_color = cv2.applyColorMap(anomaly_map_gray, cv2.COLORMAP_JET)

    # 拼接图 (横向)
    img_concat = np.hstack([defect, target, anomaly_map_color])

    # 保存，保持原本尺寸
    cv2.imwrite(save_path + '_concat.png', img_concat)

def restore_from_normalized_numpy(img_origin: np.ndarray, mean=None, std=None):
    """
    恢复经过 Normalize 处理过的图像 (基于numpy，不转tensor)。
    
    参数:
        img_origin (np.ndarray): 输入图像，(H, W, C)，float32，范围[0, 1]
        mean (list or tuple): 通道均值，例如 [0.485, 0.456, 0.406]
        std (list or tuple): 通道标准差，例如 [0.229, 0.224, 0.225]
    
    返回:
        np.ndarray: 反归一化后的图像，(H, W, C)，float32，范围[0, 1]
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = np.array(mean).reshape(1, 1, 3)  # 适配 (H, W, C)
    std = np.array(std).reshape(1, 1, 3)
    
    # 反归一化
    img_restored = img_origin * std + mean

    # 保证数值在[0,1]之间
    img_restored = np.clip(img_restored, 0.0, 1.0)

    # 转化到[0,255]之间
    img_restored = (img_restored * 255).astype(np.uint8)

    return img_restored


def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, save_dir=None, dataset_root=None, normalize_amap=False):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    anomaly_map_list = []
    gt_mask_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    os.makedirs(save_dir, exist_ok=True)  # 创建根目录

    with torch.no_grad():
        for img, gt, label, img_path_batch in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map_batch, _ = cal_anomaly_maps(en, de, img.shape[-1])
            # anomaly_map_batch = output[4]

            if resize_mask is not None:
                anomaly_map_batch = F.interpolate(anomaly_map_batch, size=resize_mask, mode='bilinear', align_corners=True)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
                img = F.interpolate(img, size=resize_mask, mode='bilinear', align_corners=True)

            anomaly_map_batch = gaussian_kernel(anomaly_map_batch)

            if normalize_amap:
                # ======= 你的归一化方式 =======
                scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    scores.append(anomaly_map)

                scores = np.array(scores)  # (B,H,W)
                min_scores = np.min(scores)
                max_scores = np.max(scores)
                anomaly_scores = (scores - min_scores) / (max_scores - min_scores + 1e-10)
                anomaly_scores = np.clip(anomaly_scores, 0, 1)
            else:
                # ======= 直接用原始 anomaly map =======
                anomaly_scores = []
                for idx in range(anomaly_map_batch.shape[0]):
                    anomaly_map = anomaly_map_batch[idx, 0].detach().cpu().numpy()
                    anomaly_map = np.clip(anomaly_map, 0, 1)
                    anomaly_scores.append(anomaly_map)
                anomaly_scores = np.array(anomaly_scores)

            # 保存每一张 anomaly map 和可视化
            for idx, img_path in enumerate(img_path_batch):
                anomaly_map = anomaly_scores[idx]

            #     relative_path = os.path.relpath(img_path, start=dataset_root)
            #     save_path = os.path.join(save_dir, relative_path)
            #     save_path = save_path.replace('.png', '')
            #     save_dirname = os.path.dirname(save_path)
            #     os.makedirs(save_dirname, exist_ok=True)

            #     img_origin = img[idx].detach().cpu().numpy()
            #     img_origin = np.transpose(img_origin, (1,2,0))
            #     img_origin = restore_from_normalized_numpy(img_origin)
            #     if img_origin.shape[2] == 1:
            #         img_origin = cv2.cvtColor(img_origin, cv2.COLOR_GRAY2BGR)
            #     else:
            #         img_origin = cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR)

            #     # target
                gt_mask = gt[idx, 0].detach().cpu().numpy()  # (H,W)
                gt_mask = (gt_mask * 255).astype(np.uint8)
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

                # anomaly_map_gray
                anomaly_map_gray = (anomaly_map * 255).astype(np.uint8)
                
                anomaly_map_list.append(anomaly_map_gray)
                gt_mask_list.append(gt_mask[:,:,0])

            #     # 调用保存函数
            #     save_concat_image(img_origin, gt_mask, anomaly_map_gray, save_path)


            # ======= 继续后续正常处理 batch 数据 =======
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(torch.tensor(anomaly_scores).unsqueeze(1).to(device))  # (B,1,H,W)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(torch.tensor(anomaly_scores).flatten(1).to(device), dim=1)[0]
            else:
                anomaly_map_flat = torch.tensor(anomaly_scores).flatten(1).to(device)
                sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, anomaly_map_list, gt_mask_list]

def evaluation_batch_vis_ZS(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None, save_root=None):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            _ = model(img)
            anomaly_map = model.distance
            side = int(model.distance.shape[1]**0.5)
            anomaly_map = anomaly_map.reshape([anomaly_map.shape[0], side, side]).contiguous()
            anomaly_map = torch.unsqueeze(anomaly_map, dim=1)
            anomaly_map = F.interpolate(anomaly_map, size=img.shape[-1], mode='bilinear', align_corners=True)
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            anomaly_map = gaussian_kernel(anomaly_map)

            save_imag_ZS(img, anomaly_map, gt, model.distance, save_root, img_path)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            # gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

         # GPU acceleration
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)

        # Only CPU
        # aupro_px = compute_pro(gt_list_px, pr_list_px)
        # gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
        # auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        # auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        # ap_px = average_precision_score(gt_list_px, pr_list_px)
        # ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
        # f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        # f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
        

COLORS = np.array([0, 0, 0, 0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0]).astype(np.float32).reshape(-1, 3)


def save_dependencies_files(root_path):
    abs_cwd = os.path.abspath(os.getcwd())
    # 获取当前加载的所有模块
    modules = list(sys.modules.values())
    # 获取所有依赖的文件的路径
    dependency_files = []
    for module in modules:
        try:
            file_path = os.path.abspath(inspect.getfile(module))
            if os.path.isfile(file_path) and os.path.commonpath([abs_cwd, file_path]) == abs_cwd:
                dependency_files.append(file_path)
        except TypeError:
            # 忽略无法获取文件路径的模块
            continue
    dependency_files = list(set(dependency_files))
    for file_path in dependency_files:
        save_path = os.path.join(root_path, os.path.relpath(file_path, abs_cwd))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        shutil.copy(file_path, save_path)
    return dependency_files

def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return lambda worker_id: fix_seeds(seed+worker_id)
    
def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
