#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate AASCE train/val JSON center and scale so that affine-cropped inputs
cover all 68 spine landmarks.

Why this is needed:
HRNet/RDPose get_affine_transform effectively uses scale[0] as the source width,
and the source height is determined by IMAGE_SIZE aspect ratio. If scale[0] is
too small, top/bottom vertebrae may be transformed outside the 256x512 input.

Default behavior:
  input : <root>/annot/train.json and <root>/annot/val.json
  output: <root>/annot/train_fullcrop.json and <root>/annot/val_fullcrop.json

You can then train with:
  DATASET.TRAIN_SET train_fullcrop
  DATASET.TEST_SET val_fullcrop
"""

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate AASCE center/scale from 68 landmarks."
    )
    parser.add_argument("--root", default="../data/aasce", type=str,
                        help="AASCE data root, e.g. ../data/aasce")
    parser.add_argument("--annot-dir", default="annot", type=str,
                        help="Annotation directory under root.")
    parser.add_argument("--train-name", default="train", type=str,
                        help="Input train annotation name without .json")
    parser.add_argument("--val-name", default="val", type=str,
                        help="Input val annotation name without .json")
    parser.add_argument("--suffix", default="_fullcrop", type=str,
                        help="Output suffix. Default creates train_fullcrop.json and val_fullcrop.json")
    parser.add_argument("--image-size", default="256,512", type=str,
                        help="Network input size W,H. Default: 256,512")
    parser.add_argument("--padding", default=1.25, type=float,
                        help="Padding ratio around landmark bbox. Recommended: 1.20~1.35")
    parser.add_argument("--pixel-std", default=200.0, type=float,
                        help="HRNet pixel std. Default: 200")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite train.json/val.json after backing up originals.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report statistics, do not save JSON files.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_center_scale_from_joints(joints, joints_vis, image_size, padding=1.25, pixel_std=200.0):
    """
    Compute HRNet-style center/scale from visible landmarks.

    image_size: [W, H], e.g. [256, 512]
    aspect_ratio = W / H

    We first compute landmark bbox, then expand the bbox to match the input aspect ratio:
      if bbox is too tall, increase width;
      if bbox is too wide, increase height.

    Then apply padding.
    """
    joints = np.asarray(joints, dtype=np.float32)
    joints_vis = np.asarray(joints_vis)

    if joints_vis.ndim == 2:
        visible = joints_vis[:, 0] > 0
    else:
        visible = joints_vis > 0

    # Fallback: if visibility is unavailable, use all points.
    if visible.sum() == 0:
        visible = np.ones((joints.shape[0],), dtype=bool)

    pts = joints[visible, :2]

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    w = float(xmax - xmin)
    h = float(ymax - ymin)

    # Avoid degenerate boxes.
    w = max(w, 1.0)
    h = max(h, 1.0)

    c_x = float((xmin + xmax) * 0.5)
    c_y = float((ymin + ymax) * 0.5)

    image_w, image_h = image_size
    aspect_ratio = float(image_w) / float(image_h)

    # Make bbox aspect ratio consistent with model input.
    # For 256x512, aspect_ratio = 0.5, so height should be 2 * width.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # Add padding to ensure all landmarks are safely inside after affine transform.
    w *= padding
    h *= padding

    center = [c_x, c_y]
    scale = [w / pixel_std, h / pixel_std]

    return center, scale


def affine_matrix_hrnet(center, scale, image_size, pixel_std=200.0):
    """
    A lightweight reimplementation of HRNet get_affine_transform for rot=0.

    For rot=0, x and y mapping can be expressed simply:
      x_out = (x - (cx - src_w/2)) * dst_w / src_w
      y_out = (y - (cy - src_h/2)) * dst_h / src_h

    Here src_w = scale[0] * pixel_std
         src_h = src_w * dst_h / dst_w

    Because we set scale[0]/scale[1] according to image aspect, this should
    match the intended crop extent. This is used only for verification.
    """
    dst_w, dst_h = float(image_size[0]), float(image_size[1])

    src_w = float(scale[0]) * pixel_std
    src_h = src_w * dst_h / dst_w

    cx, cy = float(center[0]), float(center[1])

    x0 = cx - src_w * 0.5
    y0 = cy - src_h * 0.5

    return x0, y0, src_w, src_h


def count_inside_after_crop(joints, center, scale, image_size, pixel_std=200.0):
    joints = np.asarray(joints, dtype=np.float32)
    dst_w, dst_h = float(image_size[0]), float(image_size[1])

    x0, y0, src_w, src_h = affine_matrix_hrnet(center, scale, image_size, pixel_std)

    x = (joints[:, 0] - x0) * dst_w / src_w
    y = (joints[:, 1] - y0) * dst_h / src_h

    inside = (x >= 0) & (x < dst_w) & (y >= 0) & (y < dst_h)
    return int(inside.sum()), np.stack([x, y], axis=1), inside


def process_split(records, split_name, image_size, padding, pixel_std):
    new_records = []
    rows = []

    old_inside_counts = []
    new_inside_counts = []

    for idx, rec in enumerate(records):
        rec_new = copy.deepcopy(rec)

        joints = np.asarray(rec["joints"], dtype=np.float32)
        joints_vis = np.asarray(rec.get("joints_vis", np.ones((len(joints),), dtype=np.float32)))

        old_center = rec.get("center", None)
        old_scale = rec.get("scale", None)

        if old_center is not None and old_scale is not None:
            old_inside, _, _ = count_inside_after_crop(
                joints, old_center, old_scale, image_size, pixel_std
            )
        else:
            old_inside = -1

        new_center, new_scale = get_center_scale_from_joints(
            joints,
            joints_vis,
            image_size=image_size,
            padding=padding,
            pixel_std=pixel_std
        )

        new_inside, trans_joints, inside_mask = count_inside_after_crop(
            joints, new_center, new_scale, image_size, pixel_std
        )

        rec_new["center"] = [float(new_center[0]), float(new_center[1])]
        rec_new["scale"] = [float(new_scale[0]), float(new_scale[1])]

        new_records.append(rec_new)

        old_inside_counts.append(old_inside)
        new_inside_counts.append(new_inside)

        rows.append({
            "split": split_name,
            "index": idx,
            "image": rec.get("image", ""),
            "old_inside": old_inside,
            "new_inside": new_inside,
            "old_center_x": old_center[0] if old_center is not None else "",
            "old_center_y": old_center[1] if old_center is not None else "",
            "old_scale_x": old_scale[0] if old_scale is not None else "",
            "old_scale_y": old_scale[1] if old_scale is not None else "",
            "new_center_x": new_center[0],
            "new_center_y": new_center[1],
            "new_scale_x": new_scale[0],
            "new_scale_y": new_scale[1],
            "outside_indices_after_new": " ".join([str(i) for i, ok in enumerate(inside_mask) if not ok])
        })

    old_valid = [x for x in old_inside_counts if x >= 0]

    print("\n===== {} =====".format(split_name))
    if len(old_valid) > 0:
        print("old inside landmarks: min={}, mean={:.2f}, max={}".format(
            min(old_valid), float(np.mean(old_valid)), max(old_valid)
        ))
    print("new inside landmarks: min={}, mean={:.2f}, max={}".format(
        min(new_inside_counts), float(np.mean(new_inside_counts)), max(new_inside_counts)
    ))
    print("samples with all 68 inside after new crop: {}/{}".format(
        sum([x == 68 for x in new_inside_counts]), len(new_inside_counts)
    ))

    return new_records, rows


def backup_file(path):
    path = Path(path)
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        i = 1
        while True:
            backup = path.with_suffix(path.suffix + ".bak{}".format(i))
            if not backup.exists():
                break
            i += 1
    backup.write_bytes(path.read_bytes())
    return backup


def main():
    args = parse_args()

    image_size = [int(x) for x in args.image_size.split(",")]
    assert len(image_size) == 2, "--image-size should be W,H, e.g. 256,512"

    root = Path(args.root)
    annot_dir = root / args.annot_dir

    train_path = annot_dir / (args.train_name + ".json")
    val_path = annot_dir / (args.val_name + ".json")

    print("=> root      :", root)
    print("=> train json:", train_path)
    print("=> val json  :", val_path)
    print("=> image size:", image_size)
    print("=> padding   :", args.padding)

    train_records = load_json(train_path)
    val_records = load_json(val_path)

    new_train, rows_train = process_split(
        train_records, "train", image_size, args.padding, args.pixel_std
    )
    new_val, rows_val = process_split(
        val_records, "val", image_size, args.padding, args.pixel_std
    )

    rows = rows_train + rows_val

    if args.dry_run:
        print("\n=> dry run enabled; no files saved.")
        return

    if args.overwrite:
        train_out = train_path
        val_out = val_path

        train_backup = backup_file(train_path)
        val_backup = backup_file(val_path)
        print("=> backed up train to:", train_backup)
        print("=> backed up val to  :", val_backup)
    else:
        train_out = annot_dir / (args.train_name + args.suffix + ".json")
        val_out = annot_dir / (args.val_name + args.suffix + ".json")

    save_json(new_train, train_out)
    save_json(new_val, val_out)

    report_path = annot_dir / ("center_scale_regen_report" + args.suffix + ".csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=> saved train json:", train_out)
    print("=> saved val json  :", val_out)
    print("=> saved report    :", report_path)
    print("\nUse these config overrides for training/testing:")
    if args.overwrite:
        print("  DATASET.TRAIN_SET train DATASET.TEST_SET val")
    else:
        print("  DATASET.TRAIN_SET {}{} DATASET.TEST_SET {}{}".format(
            args.train_name, args.suffix, args.val_name, args.suffix
        ))


if __name__ == "__main__":
    main()
