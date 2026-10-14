# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def _get_test_cfg(config, name, default):
    if hasattr(config.TEST, name):
        return getattr(config.TEST, name)
    return default


def _fit_spine_curve_x_from_y(centers, degree, threshold, max_iter):
    valid = np.isfinite(centers).all(axis=1)
    if valid.sum() < 4:
        return None, valid

    fit_degree = min(int(degree), int(valid.sum()) - 1)
    inliers = valid.copy()

    for _ in range(max_iter):
        if inliers.sum() <= fit_degree:
            break

        y = centers[inliers, 1]
        x = centers[inliers, 0]
        try:
            coeff = np.polyfit(y, x, fit_degree)
        except np.linalg.LinAlgError:
            return None, valid

        pred_x = np.polyval(coeff, centers[:, 1])
        residual = np.abs(centers[:, 0] - pred_x)

        valid_residual = residual[valid]
        median_residual = np.median(valid_residual)
        mad = np.median(np.abs(valid_residual - median_residual))
        robust_sigma = 1.4826 * mad
        robust_threshold = max(float(threshold), 3.0 * robust_sigma)

        next_inliers = valid & (residual <= robust_threshold)
        if next_inliers.sum() <= fit_degree:
            break
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers

    if inliers.sum() <= fit_degree:
        return None, valid

    coeff = np.polyfit(centers[inliers, 1], centers[inliers, 0], fit_degree)
    return coeff, inliers


def _nearest_inlier_neighbor(centers, inliers, idx, direction):
    j = idx + direction
    while 0 <= j < len(centers):
        if inliers[j]:
            return centers[j]
        j += direction
    return None


def _spine_curve_refine_single(preds, config):
    num_landmarks = int(_get_test_cfg(config, 'SPINE_CURVE_NUM_LANDMARKS', 68))
    num_vertebrae = int(_get_test_cfg(config, 'SPINE_CURVE_NUM_VERTEBRAE', 17))
    points_per_vertebra = int(
        _get_test_cfg(config, 'SPINE_CURVE_POINTS_PER_VERTEBRA', 4)
    )

    if preds.shape[0] < num_landmarks:
        return preds
    if num_landmarks != num_vertebrae * points_per_vertebra:
        return preds

    refined = preds.copy()
    landmarks = refined[:num_landmarks, :2]
    centers = landmarks.reshape(num_vertebrae, points_per_vertebra, 2).mean(axis=1)

    center_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    center_steps = center_steps[np.isfinite(center_steps)]
    if center_steps.size == 0:
        return preds

    threshold_factor = float(
        _get_test_cfg(config, 'SPINE_CURVE_THRESHOLD_FACTOR', 0.75)
    )
    min_threshold = float(
        _get_test_cfg(config, 'SPINE_CURVE_MIN_THRESHOLD', 12.0)
    )
    threshold = max(min_threshold, threshold_factor * float(np.median(center_steps)))

    coeff, inliers = _fit_spine_curve_x_from_y(
        centers,
        _get_test_cfg(config, 'SPINE_CURVE_POLY_DEGREE', 6),
        threshold,
        int(_get_test_cfg(config, 'SPINE_CURVE_MAX_FIT_ITERS', 4))
    )
    if coeff is None:
        return preds

    curve_x = np.polyval(coeff, centers[:, 1])
    residual = np.abs(centers[:, 0] - curve_x)
    outliers = np.isfinite(residual) & (residual > threshold)

    target_start = int(_get_test_cfg(config, 'SPINE_CURVE_TARGET_START', 0))
    target_end = int(_get_test_cfg(config, 'SPINE_CURVE_TARGET_END', num_vertebrae - 1))
    correct_y = bool(_get_test_cfg(config, 'SPINE_CURVE_CORRECT_Y', False))

    for vertebra_idx in range(num_vertebrae):
        if vertebra_idx < target_start or vertebra_idx > target_end:
            continue
        if not outliers[vertebra_idx]:
            continue

        old_center = centers[vertebra_idx]
        left = _nearest_inlier_neighbor(centers, inliers, vertebra_idx, -1)
        right = _nearest_inlier_neighbor(centers, inliers, vertebra_idx, 1)

        if left is not None and right is not None:
            new_center = (left + right) * 0.5
        else:
            new_center = old_center.copy()
            new_center[0] = curve_x[vertebra_idx]

        if not correct_y:
            new_center[1] = old_center[1]

        delta = new_center - old_center
        start = vertebra_idx * points_per_vertebra
        end = start + points_per_vertebra
        refined[start:end, :2] += delta

    return refined


def spine_curve_refine_preds(preds, config):
    if not bool(_get_test_cfg(config, 'SPINE_CURVE_POST_PROCESS', False)):
        return preds

    refined = preds.copy()
    for i in range(refined.shape[0]):
        refined[i] = _spine_curve_refine_single(refined[i], config)
    return refined


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    preds = spine_curve_refine_preds(preds, config)

    return preds, maxvals
