import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from glob import glob
from itertools import chain
from pprint import pformat
import argparse
import json

import sys
sys.path.append('./')

from loguru import logger
from tqdm import tqdm
import cv2 as cv
import numpy as np
from urllib.request import urlopen
import timm
import torch

from dataset import DATASET_INFOS, test_transform, read_image
from models import MODEL_INFOS, BaseModel
from models.feb import get_feb
from utils import save_dependencies_files, fix_seeds
from pathlib import Path

# 切换到项目根目录（假设脚本在 tools/ 下）
project_root = Path(__file__).parent.parent.resolve()
os.chdir(project_root)

@torch.no_grad()
def gen_foreground(save_path, dataset_name, model_name, layer, resize, vis):
    device = torch.device('cuda')
    logger.info(f'gen_foreground')
    logger.info(f'save to {save_path}')
    logger.info(f'params: {dataset_name} {model_name} {layer} {resize} {vis}')
    assert os.path.exists(os.path.join('data', dataset_name)), f'{dataset_name} not exists'
    dataset_info = DATASET_INFOS[dataset_name]
    for sub_category in dataset_info[1]:  # object
        fix_seeds(66)
        model: BaseModel = MODEL_INFOS[model_name]['cls']([layer], input_size=resize).to(device)
        # 修改 MODEL_INFOS 的配置，或者在调用时传递参数覆盖内部下载行为
        # 假设 MODEL_INFOS、model_name、layer、resize 和 device 都已定义

        # # ① 先创建模型，但不下载预训练权重
        # model = timm.create_model('densenet201.tv_in1k', pretrained=False).to(device)
        # model.default_cfg['shape'] = (3, resize, resize)
        #
        # # ② 指定本地权重文件路径
        # checkpoint_path = '/home/hy/xsy_pan/250411CPRmaster/checkpoints/pytorch_model.bin'
        #
        # # ③ 加载权重文件
        # state_dict = torch.load(checkpoint_path, map_location=device)
        #
        # # 如果权重文件存在 'state_dict' 键（即包含了额外的信息），请提取实际的模型参数
        # if 'state_dict' in state_dict:
        #     state_dict = state_dict['state_dict']
        #
        # # ④ 将本地权重加载到模型中
        # model.load_state_dict(state_dict)
        model.eval()
        root_dir = os.path.join('data', dataset_name, sub_category)
        logger.info(f'generate {sub_category}')
        cur_target_save_path = os.path.join(save_path, sub_category)
        os.makedirs(cur_target_save_path, exist_ok=True)
        train_image = {}
        train_ks = []
        train_image_fns = sorted(glob(os.path.join(root_dir, 'train/*/*')))
        train_features = torch.zeros(len(train_image_fns), *model.shapes[0][1:], device=device)
        for i, fn in enumerate(tqdm(train_image_fns, desc='extract train features', leave=False)):
            assert os.path.exists(fn), f'{fn} not exists'
            k = os.path.relpath(fn, root_dir)
            train_ks.append(k)
            image = read_image(fn, (resize, resize))
            image_t = test_transform(image)
            feature = model(image_t[None].to(device))[0]  # 使用 image_t[None] 给图片增加 batch 维度，再送入模型
            train_features[i:i+1] = feature.detach()
            if vis:
                train_image[k] = image
        logger.info('predict foreground')
        feb = get_feb(train_features).to(device).eval()  # get_feb(train_features) 根据训练集的特征计算得到一个前景提取模型（FEB）
        for fn in tqdm(sorted(glob(os.path.join(root_dir, 'train/*/*'))) + sorted(glob(os.path.join(root_dir, 'test/*/*'))), desc='predict data', leave=False):
            assert os.path.exists(fn), f'{fn} not exists'
            k = os.path.relpath(fn, root_dir)
            image = read_image(fn, (resize, resize))
            image_t = test_transform(image)
            feature = model(image_t[None].to(device))[0]
            foreground = feb(feature)[0, 0].cpu().numpy() # 利用 FEB 模型对提取的特征进行处理，生成前景掩码
            # save
            cur_save_dir = os.path.dirname(os.path.join(cur_target_save_path, k))
            os.makedirs(cur_save_dir, exist_ok=True)
            cur_image_name = os.path.basename(k).split('.', 1)[0]
            if vis:
                cv.imwrite(os.path.join(cur_save_dir, f'f_{cur_image_name}.png'), (foreground*255.).astype(np.uint8))
            np.save(os.path.join(cur_save_dir, f'f_{cur_image_name}.npy'), foreground)
        torch.save(feb, os.path.join(cur_target_save_path, f'feb.pth'))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("-lp", "--log-path", type=str, default=None, help="log path")
    # data
    parser.add_argument("--dataset-name", type=str, default="mvtec_ad_2_vial_fruit")
    parser.add_argument("--resize", type=int, default=320, help="image resize")
    # vis
    parser.add_argument("--vis", action="store_true", help='save vis result')
    # model
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    parser.add_argument("--layer", type=str, default='features.denseblock1', choices=list(chain(*[v['layers']for k, v in MODEL_INFOS.items()])), help=f'feature layer, ' + ", ".join([f"{k}: {v['layers']}" for k, v in MODEL_INFOS.items()]))
    args = parser.parse_args()
    # check
    if args.layer not in MODEL_INFOS[args.pretrained_model]['layers']:
        parser.error(f'{args.layer} not in {MODEL_INFOS[args.pretrained_model]["layers"]}')
    if args.log_path is None:
        args.log_path = f'log/foreground/foreground_{args.dataset_name}'

    logger.add(os.path.join(args.log_path, 'runtime.log'))
    logger.info('args: \n' + pformat(vars(args)))
    assert torch.cuda.is_available(), f'cuda is not available'
    save_dependencies_files(os.path.join(args.log_path, 'src'))
    gen_foreground(args.log_path, args.dataset_name, args.pretrained_model, args.layer, args.resize, args.vis)
    