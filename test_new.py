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
from pathlib import Path
from tools.convert_tiff_to_float16 import convert_images_to_float16_tiff_and_cleanup, convert_images_to_float16_tiff_only

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

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="mvtecnew_test")
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



def to_result_path(image_fn: str) -> Path:
    """
    将
      ./data/<dataset>/<category>/test/<subset>/<filename>
    转成
      ./result/<dataset>/anomaly_images/<category>/<subset>/<filename>
    """
    p = Path(image_fn)
    parts = p.parts
    # 找到 'data' 在 parts 中的索引
    try:
        idx = parts.index('data')
    except ValueError:
        raise ValueError(f"Path does not contain 'data' segment: {image_fn}")

    # 提取 dataset, category, subset, filename
    dataset  = parts[idx + 1]        # mvtec_test_vial_fruit
    category = parts[idx + 2]        # vial
    # parts[idx+3] 应该是 'test'
    subset   = parts[idx + 4]        # test_private
    filename = p.name[:-4]+".tiff"                # 000_regular.png

    # 拼接新路径
    new_path = Path('results') / 'anomaly_images' / category / subset / filename
    return new_path

def to_bin_result_path(image_fn: str) -> Path:
    """
    将
      ./data/<dataset>/<category>/test/<subset>/<filename>
    转成
      ./result/<dataset>/anomaly_images/<category>/<subset>/<filename>
    """
    p = Path(image_fn)
    parts = p.parts
    # 找到 'data' 在 parts 中的索引
    try:
        idx = parts.index('data')
    except ValueError:
        raise ValueError(f"Path does not contain 'data' segment: {image_fn}")

    # 提取 dataset, category, subset, filename
    dataset  = parts[idx + 1]        # mvtec_test_vial_fruit
    category = parts[idx + 2]        # vial
    # parts[idx+3] 应该是 'test'
    subset   = parts[idx + 4]        # test_private
    filename = p.name                # 000_regular.png

    # 拼接新路径
    new_path = Path('results') / dataset / 'anomaly_images_threshold' / category / subset / filename
    return new_path

@torch.no_grad()
def test(model: CPR, train_fns, test_fns, retrieval_result, foreground_result, resize, region_sizes, root_dir, knn, T, sub_category):
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
        # image = read_image(image_fn, (resize, resize))
        # 读取图像
        image = read_image(image_fn)  # image shape: [C, H, W]

        # 获取原始尺寸
        original_height, original_width = image.shape[0], image.shape[1]

        # 调整图像大小
        image = read_image(image_fn, (resize, resize))

        image_t = test_transform(image)
        k = os.path.relpath(image_fn, root_dir)
        image_name = os.path.basename(k)[:-4]
        anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
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
            size=(image_t.shape[1], image_t.shape[2]),
            mode="bilinear", align_corners=False
        )[0, 0]
        if k in foreground_result:
            foreground_weight = torch.from_numpy(
                cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
            foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
            score = score * foreground_weight
        score_g = gaussian_blur(score[None], (33, 33), 4)[0]  # PatchCore
        score_out = F.interpolate(
            score[None, None],
            size=(original_height, original_width),
            mode="bilinear", align_corners=False
        )[0, 0]
        # 计算最小值和最大值
        # score_out_np = score_out.detach().cpu().numpy()
        # score_out_clip = np.clip(anomaly_map, 0, 1)

        det_score = torch.topk(score_g.flatten(), k=T)[0].sum()  # DeSTSeg
        preds['i'].append(det_score)
        preds['p'].append(score_g)
        # —— 新增：保存原始异常图 ——
        # 1) 转到 CPU，转为 numpy
        anomaly_map = score_out.detach().cpu().numpy()
        pred_np_clipped = np.clip(anomaly_map, 0, 1)
        # # 2) 归一化到 [0,1]
        # norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        # 3) 转为 0-255 灰度
        gray = (pred_np_clipped * 255).astype(np.uint8)
        # 4) 构造保存路径，并创建目录
        # save_path = image_fn.replace(os.sep + 'data' + os.sep, os.sep + 'result' + os.sep)
        save_path = to_result_path(image_fn)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, gray)

    #     if sub_category == 'fruit_jelly':
    #         _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)  # T=127 :contentReference[oaicite:0]{index=0}
    #     elif sub_category =='vial':
    #         if anomaly_name == 'test_private':
    #             _, binary = cv2.threshold(gray, 164, 255, cv2.THRESH_BINARY)  # T=127 :contentReference[oaicite:0]{index=0}
    #         else:
    #             _, binary = cv2.threshold(gray, 177, 255,
    #                                       cv2.THRESH_BINARY)  # T=127 :contentReference[oaicite:0]{index=0}

    #     save_bin = to_bin_result_path(image_fn)
    #     os.makedirs(os.path.dirname(save_bin), exist_ok=True)
    #     cv2.imwrite(save_bin, binary)
    # parts = save_path.parts
    # save_tiff = Path(*parts[:3])
    # convert_images_to_float16_tiff_only(str(save_tiff))



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
        test(model, train_fns, test_fns, retrieval_result, foreground_result, args.resize, args.region_sizes, root_dir, args.k_nearest, args.T, sub_category)



# def seed_everything(seed: int = 42):
#     # Python、NumPy
#     random.seed(seed)
#     np.random.seed(seed)
#     # PyTorch CPU & GPU
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     # 禁用 CuDNN 的自动调优，以确保确定性
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark     = False

if __name__ == "__main__":
    parser = get_args_parser()
    # seed_everything(66)
    args = parser.parse_args()
    main(args)