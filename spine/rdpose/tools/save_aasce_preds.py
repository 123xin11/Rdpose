# ------------------------------------------------------------------------------
# Save AASCE predictions to all_preds.npy for visualization/comparison.
# Works for HRNet baseline, original RDPose, Spine3/Spine5 variants as long as:
#   - dataset.aasce is registered
#   - model can be built by models.<MODEL.NAME>.get_pose_net()
#   - TEST_SET points to val/test split
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# Make it runnable from project root: python tools/save_aasce_preds.py ...
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _init_paths  # noqa: F401

from config import cfg, update_config
import dataset
import models
from core.inference import get_final_preds


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Save AASCE predictions as all_preds.npy')
    parser.add_argument('--cfg', required=True, type=str, help='experiment config file')
    parser.add_argument('--model-file', required=True, type=str, help='trained model checkpoint/final_state.pth')
    parser.add_argument('--outdir', required=True, type=str, help='output directory')
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Required by this project update_config()
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def setup_logger():
    logging.basicConfig(
        format='%(asctime)-15s %(message)s',
        level=logging.INFO
    )


def load_model(model, model_file):
    checkpoint = torch.load(model_file, map_location='cpu')

    # final_state.pth is usually a raw state_dict.
    # checkpoint.pth may contain {'state_dict': ...}.
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove optional DataParallel prefix.
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    logger.info('=> loaded model: {}'.format(model_file))
    logger.info('=> missing keys: {}'.format(len(missing)))
    logger.info('=> unexpected keys: {}'.format(len(unexpected)))
    if len(missing) > 0:
        logger.info('=> first missing keys: {}'.format(list(missing)[:10]))
    if len(unexpected) > 0:
        logger.info('=> first unexpected keys: {}'.format(list(unexpected)[:10]))

    return model


def main():
    args = parse_args()
    setup_logger()
    update_config(cfg, args)

    logger.info('=> args: {}'.format(args))
    logger.info('=> output dir: {}'.format(args.outdir))
    os.makedirs(args.outdir, exist_ok=True)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model = load_model(model, args.model_file)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    val_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    num_samples = len(val_dataset)
    num_joints = cfg.MODEL.NUM_JOINTS

    all_preds = np.zeros((num_samples, num_joints, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
    image_paths = []

    idx = 0

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            output = model(input)

            if isinstance(output, list):
                output = output[-1]

            # HRNet outputs 68; RDPose/Spine variants may output extra center heatmaps.
            output = output[:, :num_joints, :, :]

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                cfg,
                output.detach().cpu().numpy(),
                c,
                s
            )

            num_images = input.size(0)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, axis=1)
            all_boxes[idx:idx + num_images, 5] = score

            image_paths.extend(meta['image'])
            idx += num_images

            if i % 20 == 0:
                logger.info('Test: [{}/{}]'.format(i, len(val_loader)))

    all_preds_path = os.path.join(args.outdir, 'all_preds.npy')
    all_boxes_path = os.path.join(args.outdir, 'all_boxes.npy')
    image_paths_path = os.path.join(args.outdir, 'image_paths.txt')

    np.save(all_preds_path, all_preds)
    np.save(all_boxes_path, all_boxes)
    with open(image_paths_path, 'w') as f:
        for p in image_paths:
            f.write(str(p) + '\n')

    logger.info('=> saved predictions to {}'.format(all_preds_path))
    logger.info('=> saved boxes to {}'.format(all_boxes_path))
    logger.info('=> saved image paths to {}'.format(image_paths_path))

    # Also evaluate once to verify it matches your previous metrics.
    name_values, perf_indicator = val_dataset.evaluate(
        cfg,
        all_preds,
        args.outdir,
        all_boxes,
        image_paths,
        [],
        []
    )
    logger.info('=> evaluation result: {}'.format(name_values))
    logger.info('=> perf indicator: {}'.format(perf_indicator))


if __name__ == '__main__':
    main()
