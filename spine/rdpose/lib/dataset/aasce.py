from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import numpy as np
from collections import OrderedDict

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


class AASCEDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 68
        self.flip_pairs = []

        self.upper_body_ids = tuple(range(68))
        self.lower_body_ids = tuple()

        self.db = self._get_db()

        logger.info('=> load {} AASCE samples'.format(len(self.db)))

    def _get_db(self):
        ann_file = os.path.join(self.root, 'annot', self.image_set + '.json')

        with open(ann_file, 'r') as f:
            anno = json.load(f)

        gt_db = []

        for idx, a in enumerate(anno):
            joints = np.array(a['joints'], dtype=np.float32)
            joints_vis = np.array(a['joints_vis'], dtype=np.float32)

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)

            joints_3d[:, 0:2] = joints[:, 0:2]
            joints_3d_vis[:, 0] = joints_vis
            joints_3d_vis[:, 1] = joints_vis
            joints_3d_vis[:, 2] = joints_vis

            gt_db.append({
                'image': os.path.join(self.root, 'images', a['image']),
                'center': np.array(a['center'], dtype=np.float32),
                'scale': np.array(a['scale'], dtype=np.float32),
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': a['image'],
                'imgnum': idx,
                'score': 1,
            })

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        ann_file = os.path.join(self.root, 'annot', self.image_set + '.json')

        with open(ann_file, 'r') as f:
            anno = json.load(f)

        gts = []
        masks = []

        for a in anno:
            gts.append(np.array(a['joints'], dtype=np.float32))
            masks.append(np.array(a['joints_vis'], dtype=np.float32))

        gts = np.stack(gts, axis=0)
        masks = np.stack(masks, axis=0)

        pred_xy = preds[:, :, 0:2]

        error = np.linalg.norm(pred_xy - gts, axis=2)
        visible_error = error * masks

        mre = visible_error.sum() / (masks.sum() + 1e-6)

        pck_5 = ((error <= 5) * masks).sum() / (masks.sum() + 1e-6) * 100
        pck_10 = ((error <= 10) * masks).sum() / (masks.sum() + 1e-6) * 100
        pck_20 = ((error <= 20) * masks).sum() / (masks.sum() + 1e-6) * 100

        name_value = OrderedDict([
            ('MRE', mre),
            ('PCK@5px', pck_5),
            ('PCK@10px', pck_10),
            ('PCK@20px', pck_20),
        ])

        perf_indicator = 1.0 / (mre + 1e-6)

        return name_value, perf_indicator