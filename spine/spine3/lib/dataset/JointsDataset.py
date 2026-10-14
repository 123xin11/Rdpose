# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        # print(image_file)
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']


        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        # First transform the original annotated landmarks.
        # Extra center landmarks must be generated after affine transform,
        # otherwise their coordinates would stay in original-image space.
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # Add center landmarks for RDPose after affine transform.
        # AASCE spine-aware version: 3 segment centers.
        # Other datasets keep the original single global center.
        if self.num_joints == 68:
            centers, centers_vis = self._generate_aasce_segment_centers(joints, joints_vis)
            joints = np.concatenate((joints, centers), axis=0)
            joints_vis = np.concatenate((joints_vis, centers_vis), axis=0)
        else:
            center = np.zeros((1, 3), dtype=np.float32)

            vis = (joints_vis[:self.num_joints, 0] > 0).astype(np.float32)
            if np.sum(vis) > 0:
                center[:, :2] = np.sum(
                    joints[:self.num_joints, :2] * vis[:, np.newaxis],
                    axis=0
                ) / np.sum(vis)
                center[:, 2] = 2

            center_vis = np.zeros((1, 3), dtype=np.float32)
            if np.sum(vis) > 0:
                center_vis[:, :] = 1

            joints = np.concatenate((joints, center), axis=0)
            joints_vis = np.concatenate((joints_vis, center_vis), axis=0)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def _get_extra_center_num(self):
        """
        Original RDPose uses 1 global center.
        AASCE spine-aware version uses 3 segment centers.
        """
        if self.num_joints == 68:
            return 3
        return 1

    


    def _generate_aasce_segment_centers(self, joints, joints_vis):
        centers = np.zeros((3, 3), dtype=np.float32)
        centers_vis = np.zeros((3, 3), dtype=np.float32)

        # upper:  vertebra 0-5  -> landmarks 0-23
        # middle: vertebra 6-11 -> landmarks 24-47
        # lower:  vertebra 12-16 -> landmarks 48-67

        img_w = float(self.image_size[0])
        img_h = float(self.image_size[1])

        segments = [
            list(range(0, 24)),
            list(range(24, 48)),
            list(range(48, 68)),
        ]

        for sid, ids in enumerate(segments):
            ids = np.array(ids, dtype=np.int32)
            pts = joints[ids, 0:2]
            vis = joints_vis[ids, 0] > 0
            # only use landmarks inside the network input image
            inside = (
                (pts[:, 0] >= 0) & (pts[:, 0] < img_w) &
                (pts[:, 1] >= 0) & (pts[:, 1] < img_h)
            )

            valid = vis & inside

            if np.sum(valid) > 0:
                center_xy = np.mean(pts[valid], axis=0)
                centers[sid, 0:2] = center_xy
                centers[sid, 2] = 2

                centers_vis[sid, :] = 1
            else:
                # no valid landmark in this segment after affine transform
                # keep this center invisible, so it will not contribute to loss
                centers[sid, :] = 0
                centers_vis[sid, :] = 0
        return centers, centers_vis





    def _get_aasce_endpoint_target_weight(self, num_total_joints):
        """
        No endpoint weighted loss.

        This keeps AASCE target weights unchanged. It is used for the
        Spine3-Aux no-EW setting and endpoint refinement experiments.
        """
        return np.ones((num_total_joints, 1), dtype=np.float32)


    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_total_joints, 3]
                         AASCE spine-aware RDPose uses
                         68 landmarks + 3 segment centers = 71.
                         Original RDPose datasets use num_joints + 1 center.
        :param joints_vis: [num_total_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_total_joints = joints.shape[0]

        target_weight = np.ones((num_total_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        # Endpoint weighted loss for AASCE.
        # This keeps invisible joints at 0 and increases supervision strength
        # only for visible endpoint/corner landmarks.
        if self.num_joints == 68:
            endpoint_weight = self._get_aasce_endpoint_target_weight(num_total_joints)
            target_weight = target_weight * endpoint_weight

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((num_total_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3
            feat_stride = self.image_size / self.heatmap_size

            for joint_id in range(num_total_joints):
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0
                    continue

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2

                # The gaussian is not normalized; center value equals 1
                g = np.exp(
                    -((x - x0) ** 2 + (y - y0) ** 2) /
                    (2 * self.sigma ** 2)
                )

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]

                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise ValueError('Unsupported target type: {}'.format(self.target_type))

        if self.use_different_joints_weight:
            joints_weight = np.asarray(self.joints_weight, dtype=np.float32)

            if joints_weight.size == 1:
                target_weight = target_weight * float(joints_weight)
            elif joints_weight.shape[0] == target_weight.shape[0]:
                target_weight = np.multiply(target_weight, joints_weight)
            elif joints_weight.shape[0] == self.num_joints and \
                    target_weight.shape[0] > self.num_joints:
                full_weight = np.ones_like(target_weight)
                full_weight[:self.num_joints] = joints_weight
                target_weight = np.multiply(target_weight, full_weight)
            else:
                logger.warning(
                    'Skip USE_DIFFERENT_JOINTS_WEIGHT because joints_weight shape {} '
                    'does not match target_weight shape {}'.format(
                        joints_weight.shape, target_weight.shape
                    )
                )

        return target, target_weight
