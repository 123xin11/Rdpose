#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an offline augmented AASCE dataset.

The script expands only the training split:
  original + geometric + intensity + geometric/intensity.

It also writes HRNet/RDPose-style JSON annotations, so training can use:
  DATASET.ROOT <out-root> DATASET.TRAIN_SET train_aug4x DATASET.TEST_SET val_aug4x
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create AASCE augmented dataset")
    parser.add_argument("--src-root", default="AASCE2019", type=str)
    parser.add_argument("--out-root", default="AASCE2019_aug4x", type=str)
    parser.add_argument("--seed", default=20260705, type=int)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--rot", default=15.0, type=float)
    parser.add_argument("--scale-min", default=0.85, type=float)
    parser.add_argument("--scale-max", default=1.20, type=float)
    parser.add_argument("--translate", default=0.05, type=float)
    parser.add_argument("--brightness-min", default=0.8, type=float)
    parser.add_argument("--brightness-max", default=1.2, type=float)
    parser.add_argument("--contrast-min", default=0.8, type=float)
    parser.add_argument("--contrast-max", default=1.3, type=float)
    parser.add_argument("--gamma-min", default=0.8, type=float)
    parser.add_argument("--gamma-max", default=1.2, type=float)
    parser.add_argument("--noise-std", default=4.0, type=float)
    parser.add_argument("--blur-prob", default=0.25, type=float)
    parser.add_argument("--padding", default=1.25, type=float)
    parser.add_argument("--image-size", default="256,512", type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_float_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")])
    return rows


def write_float_rows(path, rows, fmt="{:.8g}"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([fmt.format(float(x)) for x in row])


def split_landmarks(row):
    arr = np.asarray(row, dtype=np.float32)
    if arr.size != 136:
        raise ValueError("Expected 136 landmark values, got {}".format(arr.size))
    # AASCE landmarks.csv stores all x coordinates first, then all y coordinates.
    return np.stack([arr[:68], arr[68:]], axis=1)


def flatten_landmarks(points_norm):
    # Keep the original AASCE CSV layout: x1..x68,y1..y68.
    return np.concatenate([points_norm[:, 0], points_norm[:, 1]]).tolist()


def apply_affine_to_points(points_xy, matrix):
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([points_xy.astype(np.float32), ones], axis=1)
    return hom.dot(matrix.T)


def random_affine(image, points_xy, args):
    h, w = image.shape[:2]
    angle = random.uniform(-args.rot, args.rot)
    scale = random.uniform(args.scale_min, args.scale_max)
    tx = random.uniform(-args.translate, args.translate) * w
    ty = random.uniform(-args.translate, args.translate) * h

    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale).astype(np.float32)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    aug = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    pts = apply_affine_to_points(points_xy, matrix)
    return aug, pts, {"angle": angle, "scale_aug": scale, "tx": tx, "ty": ty}


def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def random_intensity(image, args):
    brightness = random.uniform(args.brightness_min, args.brightness_max)
    contrast = random.uniform(args.contrast_min, args.contrast_max)
    gamma = random.uniform(args.gamma_min, args.gamma_max)

    out = image.astype(np.float32)
    out = (out - 127.5) * contrast + 127.5
    out = np.clip(out * brightness, 0, 255).astype(np.uint8)
    out = adjust_gamma(out, gamma)

    noise_std = random.uniform(0.0, args.noise_std)
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    blur = False
    if random.random() < args.blur_prob:
        out = cv2.GaussianBlur(out, (3, 3), 0)
        blur = True

    return out, {
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "noise_std": noise_std,
        "blur": blur,
    }


def valid_points(points_xy, width, height, min_inside=60):
    inside = (
        (points_xy[:, 0] >= 0) & (points_xy[:, 0] < width) &
        (points_xy[:, 1] >= 0) & (points_xy[:, 1] < height)
    )
    return int(inside.sum()) >= min_inside


def make_augmented_geometric(image, points_xy, args):
    h, w = image.shape[:2]
    last = None
    for _ in range(12):
        aug, pts, meta = random_affine(image, points_xy, args)
        last = (aug, pts, meta)
        if valid_points(pts, w, h):
            return last
    return last


def compute_center_scale(points_xy, image_size, padding=1.25, pixel_std=200.0):
    pts = np.asarray(points_xy, dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    bw = max(float(xmax - xmin), 1.0)
    bh = max(float(ymax - ymin), 1.0)
    cx = float((xmin + xmax) * 0.5)
    cy = float((ymin + ymax) * 0.5)

    aspect_ratio = float(image_size[0]) / float(image_size[1])
    if bw > aspect_ratio * bh:
        bh = bw / aspect_ratio
    elif bw < aspect_ratio * bh:
        bw = bh * aspect_ratio

    bw *= padding
    bh *= padding
    return [cx, cy], [bw / pixel_std, bh / pixel_std]


def make_record(image_name, points_xy, width, height, image_size, padding):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    center, scale = compute_center_scale(points_xy, image_size, padding)

    points_norm = points_xy.copy()
    points_norm[:, 0] /= float(width)
    points_norm[:, 1] /= float(height)
    points_norm = np.clip(points_norm, 0.0, 1.0)

    return {
        "image": image_name,
        "joints": points_xy.astype(float).tolist(),
        "joints_vis": [1.0] * points_xy.shape[0],
        "center": center,
        "scale": scale,
    }, flatten_landmarks(points_norm)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_clean_dir(path, overwrite=False):
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise RuntimeError("{} already exists. Use --overwrite to replace it.".format(path))
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    train_dir = src_root / "train"
    txt_dir = src_root / "train_txt"
    image_size = [int(x) for x in args.image_size.split(",")]

    filenames = read_lines(txt_dir / "filenames.csv")
    landmark_rows = read_float_rows(txt_dir / "landmarks.csv")
    angle_rows = read_float_rows(txt_dir / "angles.csv")
    if not (len(filenames) == len(landmark_rows) == len(angle_rows)):
        raise ValueError("filenames/landmarks/angles row counts do not match")

    indices = list(range(len(filenames)))
    random.shuffle(indices)
    val_count = int(round(len(indices) * args.val_ratio))
    val_indices = set(indices[:val_count])

    ensure_clean_dir(out_root, args.overwrite)
    out_images = out_root / "images"
    out_annot = out_root / "annot"
    out_txt = out_root / "train_txt"
    out_images.mkdir(parents=True, exist_ok=True)
    out_annot.mkdir(parents=True, exist_ok=True)
    out_txt.mkdir(parents=True, exist_ok=True)

    train_records, val_records = [], []
    out_filenames, out_landmarks, out_angles, report_rows = [], [], [], []
    variants = [("orig", False, False), ("geom", True, False), ("gray", False, True), ("geom_gray", True, True)]

    for idx, filename in enumerate(filenames):
        image = cv2.imread(str(train_dir / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to read image: {}".format(train_dir / filename))

        h, w = image.shape[:2]
        points_norm = split_landmarks(landmark_rows[idx])
        points_xy = points_norm.copy()
        points_xy[:, 0] *= float(w)
        points_xy[:, 1] *= float(h)

        if idx in val_indices:
            cv2.imwrite(str(out_images / filename), image)
            record, lm_row = make_record(filename, points_xy, w, h, image_size, args.padding)
            val_records.append(record)
            out_filenames.append(filename)
            out_landmarks.append(lm_row)
            out_angles.append(angle_rows[idx])
            report_rows.append({"source_index": idx, "source_image": filename, "output_image": filename, "split": "val", "variant": "orig"})
            continue

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        for variant, do_geom, do_gray in variants:
            aug_img = image.copy()
            aug_pts = points_xy.copy()
            meta = {}

            if do_geom:
                aug_img, aug_pts, geom_meta = make_augmented_geometric(aug_img, aug_pts, args)
                meta.update(geom_meta)
            if do_gray:
                aug_img, gray_meta = random_intensity(aug_img, args)
                meta.update(gray_meta)

            out_name = filename if variant == "orig" else "{}__{}{}".format(stem, variant, suffix)
            cv2.imwrite(str(out_images / out_name), aug_img)
            record, lm_row = make_record(out_name, aug_pts, w, h, image_size, args.padding)
            train_records.append(record)
            out_filenames.append(out_name)
            out_landmarks.append(lm_row)
            out_angles.append(angle_rows[idx])
            row = {"source_index": idx, "source_image": filename, "output_image": out_name, "split": "train", "variant": variant}
            row.update(meta)
            report_rows.append(row)

    save_json(out_annot / "train_aug4x.json", train_records)
    save_json(out_annot / "val_aug4x.json", val_records)
    with open(out_txt / "filenames.csv", "w", encoding="utf-8") as f:
        for name in out_filenames:
            f.write(name + "\n")
    write_float_rows(out_txt / "landmarks.csv", out_landmarks)
    write_float_rows(out_txt / "angles.csv", out_angles)

    fieldnames = sorted(set().union(*[row.keys() for row in report_rows]))
    with open(out_root / "augment_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print("=> source samples:", len(filenames))
    print("=> train records :", len(train_records))
    print("=> val records   :", len(val_records))
    print("=> output root   :", out_root)
    print("Training overrides:")
    print("  DATASET.ROOT {} DATASET.TRAIN_SET train_aug4x DATASET.TEST_SET val_aug4x".format(out_root))


if __name__ == "__main__":
    main()
