from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import _init_paths
import models
import dataset

from config import cfg, update_config
from core.inference import get_final_preds


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize AASCE predictions')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--model-file',
                        help='trained model file',
                        required=True,
                        type=str)

    parser.add_argument('--outdir',
                        help='output visualization directory',
                        default='vis_aasce',
                        type=str)

    parser.add_argument('--vis-num',
                        help='number of images to visualize',
                        default=20,
                        type=int)

    # 下面这几个是为了兼容 update_config()
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev model directory',
                        type=str,
                        default='')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(cfg, args)

    return args

def draw_points(img, pts, color, radius=3):
    for p in pts:
        x, y = int(p[0]), int(p[1])
        cv2.circle(img, (x, y), radius, color, -1)


def main():
    args = parse_args()

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    print('=> loading model from {}'.format(args.model_file))
    model.load_state_dict(torch.load(args.model_file), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    # dataset
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    count = 0

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(valid_loader):
            if count >= args.vis_num:
                break

            input = input.cuda(non_blocking=True)

            output = model(input)

            # [1, 68, H, W]
            output_np = output.cpu().numpy()

            center = meta['center'].numpy()
            scale = meta['scale'].numpy()

            preds, maxvals = get_final_preds(cfg, output_np, center, scale)

            pred_pts = preds[0]  # [68, 2]
            gt_pts = valid_dataset.db[i]['joints_3d'][:cfg.MODEL.NUM_JOINTS, :2]

            img_path = meta['image'][0]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 画 GT: 绿色
            draw_points(img, gt_pts, (0, 255, 0), radius=3)

            # 画 Pred: 红色
            draw_points(img, pred_pts, (0, 0, 255), radius=3)

            # 图例
            cv2.putText(img, 'GT: green  Pred: red', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            save_name = os.path.basename(img_path)
            save_path = os.path.join(args.outdir, save_name)
            cv2.imwrite(save_path, img)

            print('saved:', save_path)
            count += 1

    print('Done. Saved {} images to {}'.format(count, args.outdir))


if __name__ == '__main__':
    main()