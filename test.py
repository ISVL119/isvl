from collections import defaultdict
from glob import glob
from itertools import chain
from tqdm import tqdm
import argparse
import json
import os,cv2
import torch
import random

from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataset import DATASET_INFOS, read_image, read_mask, test_transform
from metrics import compute_ap_torch, compute_pixel_auc_torch, compute_pro_torch, compute_image_auc_torch
from models import create_model, MODEL_INFOS, CPR
from utils import fix_seeds
import re
from sklearn.metrics import precision_recall_curve
from skimage.filters import threshold_otsu  # pip install scikit-image

# def f1_score_max(y_true, y_score):
#     precs, recs, thrs = precision_recall_curve(y_true, y_score)
#     f1s = 2 * precs * recs / (precs + recs + 1e-7)
#     # 因为 precision_recall_curve 返回的 thresholds 比 f1 分数少1，所以去除最后一个
#     f1s = f1s[:-1]
#     return f1s.max()
def f1_score_max(y_true, y_score):
    # 计算精确率、召回率和阈值
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # 计算 F1 分数，忽略最后一个点
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-7)

    # 找到最大 F1 分数的索引
    max_f1_index = np.argmax(f1_scores)

    # 获取最大 F1 分数及其对应的阈值
    max_f1 = f1_scores[max_f1_index]
    best_threshold = thresholds[max_f1_index]

    return max_f1, best_threshold

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
    # all_gts_tensor = torch.cat([x.flatten() for x in gt_masks])
    # all_gts = all_gts_tensor.cpu().numpy().astype(np.uint8)

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


def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="mvtecnew", choices=["mvtec_own_seg_vial_mixed","mvtec_own_seg_vial","mvtec_own_seg_fruit_mixed","mvtec_own_seg_fruit","mvtec_test1","mvtec", "mvtec_3d", "btad","mvtecnew","mvtecnew_raw","mvtec_new_fruit"], help="dataset name")
    parser.add_argument("-ss", "--scales", type=int, nargs="+", help="multiscale", default=[4, 8])
    parser.add_argument("-kn", "--k-nearest", type=int, default=10, help="k nearest")
    parser.add_argument("-r", "--resize", type=int, default=320, help="image resize")
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None, help="foreground dir")
    parser.add_argument("-rd", "--retrieval-dir", type=str, default='log/retrieval_mvtec_DenseNet_features.denseblock1_320', help="retrieval dir")
    parser.add_argument("--sub-categories", type=str, nargs="+", default=None, help="sub categories", choices=list(chain(*[x[0] for x in list(DATASET_INFOS.values())])))
    parser.add_argument("--T", type=int, default=256)  # for image-level inference
    parser.add_argument("-rs", "--region-sizes", type=int, nargs="+", help="local retrieval region size", default=[3, 1])
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None, help="checkpoints")
    return parser

@torch.no_grad()
# def test(model: CPR, train_fns, test_fns, retrieval_result, foreground_result, resize, region_sizes, root_dir, knn, T):
#     model.eval()
#     train_local_features = [torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda') for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)]
#     train_foreground_weights = []
#     k2id = {}
#     for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train local features', leave=False)):
#         k = os.path.relpath(image_fn, root_dir)
#         image = read_image(image_fn, (resize, resize))
#         image_t = test_transform(image)
#         features_list, ori_features_list = model(image_t[None].cuda())
#         for i, features in enumerate(features_list):
#             train_local_features[i][idx:idx+1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
#         if k in foreground_result:
#             train_foreground_weights.append(torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda())
#         k2id[k] = idx
#     if train_foreground_weights:
#         train_foreground_weights = torch.stack(train_foreground_weights)
#
#     gts = []
#     i_gts = []
#     preds = defaultdict(list)
#     for image_fn in tqdm(test_fns, desc='predict test data', leave=False):
#         image = read_image(image_fn, (resize, resize))
#         image_t = test_transform(image)
#         k = os.path.relpath(image_fn, root_dir)
#         image_name = os.path.basename(k)[:-4]
#         anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
#         mask_fn = os.path.join(root_dir, 'ground_truth', anomaly_name, image_name + '_mask.png')
#         if os.path.exists(mask_fn):
#             mask = read_mask(mask_fn, (resize, resize))
#         else:
#             new_mask_fn = re.sub(r'(.*)_split_(\d+)_mask(\.png)$', r'\1_mask_split_\2\3', mask_fn)
#             if os.path.exists(new_mask_fn):
#                 mask = read_mask(new_mask_fn, (resize, resize))
#             else:
#                 new_mask_fn = re.sub(r'(.*)_(\d+)_(\d+)_mask(\.png)$', r'\1_mask_\2_\3\4', mask_fn)
#                 if os.path.exists(new_mask_fn):
#                     mask = read_mask(new_mask_fn, (resize, resize))
#                 else:
#                     mask = np.zeros((resize, resize))
#
#
#         gts.append((mask > 127).astype(int))
#         i_gts.append((mask > 127).sum() > 0 and 1 or 0)
#
#         features_list, ori_features_list = model(image_t[None].cuda())
#         features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in features_list]
#         retrieval_idxs = [k2id[retrieval_k] for retrieval_k in retrieval_result[k][:knn]]
#         retrieval_features_list = [train_local_features[i][retrieval_idxs] for i in range(len(features_list))]
#
#         scores = []
#         assert len(features_list) == len(retrieval_features_list) == len(region_sizes)
#         for features, retrieval_features, region_size in zip(features_list, retrieval_features_list, region_sizes):
#             unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
#             region_features = unfold(retrieval_features).reshape(retrieval_features.shape[0], retrieval_features.shape[1], -1, retrieval_features.shape[2], retrieval_features.shape[3])  # b x c x r^2 x h x w
#             dist = (1 - (features[:, :, None] * region_features).sum(1))  # b x r^2 x h x w
#             # fill position is set to a large value
#             dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2], retrieval_features.shape[3], device=retrieval_features.device)).reshape(1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
#             score = dist.min(1)[0].min(0)[0]
#             score = F.interpolate(
#                 score[None, None],
#                 size=(features_list[0].shape[2], features_list[0].shape[3]),
#                 mode="bilinear", align_corners=False
#             )[0, 0]
#             scores.append(score)
#         score = torch.stack(scores).sum(0)
#         score = F.interpolate(
#             score[None, None],
#             size=(mask.shape[0], mask.shape[1]),
#             mode="bilinear", align_corners=False
#         )[0, 0]
#         if k in foreground_result:
#             foreground_weight = torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
#             foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
#             score = score * foreground_weight
#         score_g = gaussian_blur(score[None], (33, 33), 4)[0]  # PatchCore
#         det_score = torch.topk(score_g.flatten(), k=T)[0].sum()  # DeSTSeg
#         preds['i'].append(det_score)
#         preds['p'].append(score_g)
#     gts = torch.from_numpy(np.stack(gts)).cuda()
#     return {
#         'pro': compute_pro_torch(gts, torch.stack(preds['p'])),
#         'ap': compute_ap_torch(gts, torch.stack(preds['p'])),
#         'pixel-auc': compute_pixel_auc_torch(gts, torch.stack(preds['p'])),
#         'image-auc': compute_image_auc_torch(torch.tensor(i_gts).long().cuda(), torch.stack(preds['i'])),
#     }
#

def test(model: CPR, train_fns, test_fns, retrieval_result, foreground_result, resize, region_sizes, root_dir, knn, T):
    model.eval()
    train_local_features = [torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda')
                            for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)]
    train_foreground_weights = []
    k2id = {}
    for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train local features', leave=False)):
        k = os.path.relpath(image_fn, root_dir)
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        features_list, ori_features_list = model(image_t[None].cuda())
        for i, features in enumerate(features_list):
            train_local_features[i][idx:idx + 1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
        if k in foreground_result:
             train_foreground_weights.append(
                torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda())
        k2id[k] = idx
    if train_foreground_weights:
        train_foreground_weights = torch.stack(train_foreground_weights)

    gts = []
    i_gts = []
    preds = defaultdict(list)
    for image_fn in tqdm(test_fns, desc='predict test data', leave=False):
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        k = os.path.relpath(image_fn, root_dir)
        image_name = os.path.basename(k)[:-4]
        anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
        mask_fn = os.path.join(root_dir, 'ground_truth', anomaly_name, image_name + '_mask.png')
        if os.path.exists(mask_fn):
            mask = read_mask(mask_fn, (resize, resize))
        else:
            new_mask_fn = re.sub(r'(.*)_split_(\d+)_mask(\.png)$', r'\1_mask_split_\2\3', mask_fn)
            if os.path.exists(new_mask_fn):
                mask = read_mask(new_mask_fn, (resize, resize))
            else:
                new_mask_fn = re.sub(r'(.*)_(\d+)_(\d+)_mask(\.png)$', r'\1_mask_\2_\3\4', mask_fn)
                if os.path.exists(new_mask_fn):
                    mask = read_mask(new_mask_fn, (resize, resize))
                else:
                    mask = np.zeros((resize, resize))

        # 将 mask 转换为二值标签：像素值大于127视为正例
        gts.append((mask > 127).astype(int))
        i_gts.append((mask > 127).sum() > 0 and 1 or 0)

        features_list, ori_features_list = model(image_t[None].cuda())
        features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in
                         features_list]
        retrieval_idxs = [k2id[retrieval_k] for retrieval_k in retrieval_result[k][:knn]]
        retrieval_features_list = [train_local_features[i][retrieval_idxs] for i in range(len(features_list))]

        scores = []
        assert len(features_list) == len(retrieval_features_list) == len(region_sizes)
        for features, retrieval_features, region_size in zip(features_list, retrieval_features_list, region_sizes):
            unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
            # b x c x r^2 x h x w
            region_features = unfold(retrieval_features).reshape(
                retrieval_features.shape[0], retrieval_features.shape[1], -1, retrieval_features.shape[2],
                retrieval_features.shape[3]
            )
            dist = (1 - (features[:, :, None] * region_features).sum(1))  # b x r^2 x h x w
            # fill position is set to a large value
            dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2], retrieval_features.shape[3],
                                             device=retrieval_features.device))
                           .reshape(1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
            score = dist.min(1)[0].min(0)[0]
            score = F.interpolate(
                score[None, None],
                size=(features_list[0].shape[2], features_list[0].shape[3]),
                mode="bilinear", align_corners=False
            )[0, 0]
            scores.append(score)
        score = torch.stack(scores).sum(0)
        score = F.interpolate(
            score[None, None],
            size=(mask.shape[0], mask.shape[1]),
            mode="bilinear", align_corners=False
        )[0, 0]
        if k in foreground_result:
            foreground_weight = torch.from_numpy(
                cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
            foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
            score = score * foreground_weight
        score_g = gaussian_blur(score[None], (33, 33), 4)[0]  # PatchCore
        det_score = torch.topk(score_g.flatten(), k=T)[0].sum()  # DeSTSeg
        preds['i'].append(det_score)
        preds['p'].append(score_g)
        # —— 新增：保存原始异常图 ——
        # 1) 转到 CPU，转为 numpy
        anomaly_map = score_g.cpu().numpy()
        pred_np_clipped = np.clip(anomaly_map, 0, 1)
        # # 2) 归一化到 [0,1]
        # norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        # 3) 转为 0-255 灰度
        gray = (pred_np_clipped * 255).astype(np.uint8)
        # 4) 构造保存路径，并创建目录
        # save_path = image_fn.replace(os.sep + 'data' + os.sep, os.sep + 'result' + os.sep)
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # cv2.imwrite(save_path, gray)

        # —— 新增：Otsu 自动阈值二值化并保存 ——
        # 1) 直接用灰度图 + Otsu 计算阈值
        # otsu_th = threshold_otsu(gray)  # 自动找阈值 :contentReference[oaicite:1]{index=1}
        _, binary = cv2.threshold(gray, 193, 255, cv2.THRESH_BINARY)  # T=127 :contentReference[oaicite:0]{index=0}

        # binary = (gray > otsu_th).astype(np.uint8) * 255
        # 2) 构造保存路径
        # save_bin = image_fn.replace(os.sep + 'data' + os.sep, os.sep + 'result_binary' + os.sep)
        # os.makedirs(os.path.dirname(save_bin), exist_ok=True)
        # cv2.imwrite(save_bin, binary)


    # 将 ground truth 和预测的像素分数转为 numpy 数组并展平，用于 F1 score 计算
    gts_np = np.stack(gts).flatten()  # shape: (N * H * W,)
    pred_np = torch.stack(preds['p']).cpu().numpy().flatten()
    # # 计算最小值和最大值
    # min_val = np.min(pred_np)
    # max_val = np.max(pred_np)
    #
    # # 执行最小-最大归一化
    # pred_np_normalized = (pred_np - min_val) / (max_val - min_val + 1e-8)
    # best_threshold, pixel_f1 = find_best_threshold(gts, (preds['p']))
    # pixel_f1 = f1_score_max(gts_np, pred_np)
    pred_np_clipped = np.clip(pred_np, 0, 1)
    pixel_f1,best_threshold= f1_score_max(gts_np, pred_np_clipped)
    # # 映射到 [0,255]
    # gts_255 = (gts * 255).astype(np.uint8)  # 转成 0–255 的 uint8
    # preds_p_255 = (preds['p'] * 255).astype(np.uint8)
    #
    # # 之后再调用你的阈值搜索函数
    # best_threshold, pixel_f1 = find_best_threshold(gts_255, preds_p_255)
    print(f"最佳阈值（0–255）：{best_threshold*255}, 像素级 F1：{pixel_f1:.4f}")

    gts_tensor = torch.from_numpy(np.stack(gts)).cuda()
    return {
        'pro': compute_pro_torch(gts_tensor, torch.stack(preds['p'])),
        'ap': compute_ap_torch(gts_tensor, torch.stack(preds['p'])),
        'pixel-auc': compute_pixel_auc_torch(gts_tensor, torch.stack(preds['p'])),
        'image-auc': compute_image_auc_torch(torch.tensor(i_gts).long().cuda(), torch.stack(preds['i'])),
        'f1': pixel_f1,  # 新增的 F1-score
    }


def main(args):
    all_categories, object_categories, texture_categories = DATASET_INFOS[args.dataset_name]
    sub_categories = DATASET_INFOS[args.dataset_name][0] if args.sub_categories is None else args.sub_categories
    assert all([sub_category in all_categories for sub_category in sub_categories]), f"{sub_categories} must all be in {all_categories}"
    model_info = MODEL_INFOS[args.pretrained_model]
    layers = [model_info['layers'][model_info['scales'].index(scale)] for scale in args.scales]
    for sub_category_idx, sub_category in enumerate(sub_categories):
        fix_seeds(66)
        model             = create_model(args.pretrained_model, layers).cuda()
        if args.checkpoints is not None:
            checkpoint_fn = args.checkpoints[0] if len(args.checkpoints) == 1 else args.checkpoints[sub_category_idx]
            if '{category}' in checkpoint_fn: checkpoint_fn = checkpoint_fn.format(category=sub_category)
            model.load_state_dict(torch.load(checkpoint_fn), strict=False)
        root_dir = os.path.join('./data', args.dataset_name, sub_category)
        train_fns = sorted(glob(os.path.join(root_dir, 'train/*/*')))
        test_fns = sorted(glob(os.path.join(root_dir, 'test/*/*')))
        with open(os.path.join(args.retrieval_dir, sub_category, 'r_result.json'), 'r') as f:
            retrieval_result = json.load(f)
        foreground_result = {}
        if args.foreground_dir is not None and sub_category in object_categories:
            for fn in train_fns + test_fns:
                k = os.path.relpath(fn, root_dir)
                foreground_result[k] = os.path.join(args.foreground_dir, sub_category, os.path.dirname(k), 'f_' + os.path.splitext(os.path.basename(k))[0] + '.npy')
        ret = test(model, train_fns, test_fns, retrieval_result, foreground_result, args.resize, args.region_sizes, root_dir, args.k_nearest, args.T)
        print(f'================={sub_category}=================')
        print(ret)

def seed_everything(seed: int = 42):
    # Python、NumPy
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 禁用 CuDNN 的自动调优，以确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

if __name__ == "__main__":
    parser = get_args_parser()
    seed_everything(66)
    args = parser.parse_args()
    main(args)