#!/usr/bin/env python3
"""Evaluate AASCE 68-landmark predictions in original-image pixel coordinates."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


def basename(value):
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]


def reshape_keypoints(value, expected_points):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] in (2, 3):
        pass
    elif array.ndim == 1 and array.size == expected_points * 2:
        array = array.reshape(expected_points, 2)
    elif array.ndim == 1 and array.size == expected_points * 3:
        array = array.reshape(expected_points, 3)
    else:
        raise ValueError("unsupported keypoint shape: {}".format(array.shape))

    if array.shape[0] != expected_points:
        raise ValueError(
            "expected {} keypoints, got {}".format(expected_points, array.shape[0])
        )
    return array


def records_from_json(path, expected_points, is_ground_truth):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        if "annotations" in data:
            records = data["annotations"]
        elif "predictions" in data:
            records = data["predictions"]
        else:
            raise ValueError(
                "JSON root must be a list or contain 'annotations'/'predictions'"
            )
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("JSON root must be a list or object")

    output = {}
    for index, record in enumerate(records):
        name_value = None
        for key in ("image_name", "image_file", "file_name", "image"):
            if key in record:
                name_value = record[key]
                break
        if name_value is None:
            raise ValueError("record {} has no image filename field".format(index))

        name = basename(name_value)
        if name in output:
            raise ValueError("duplicate image record: {}".format(name))

        points = reshape_keypoints(record["keypoints"], expected_points)
        xy = points[:, :2]

        if not np.isfinite(xy).all():
            raise ValueError("{} contains NaN or infinite coordinates".format(name))

        if is_ground_truth:
            if points.shape[1] == 3:
                visible = points[:, 2] > 0
            elif "joints_vis" in record:
                visible = np.asarray(record["joints_vis"]).reshape(-1) > 0
            elif "visibility" in record:
                visible = np.asarray(record["visibility"]).reshape(-1) > 0
            else:
                visible = np.ones(expected_points, dtype=bool)
            if visible.size != expected_points:
                raise ValueError("{} has invalid visibility length".format(name))
        else:
            visible = np.ones(expected_points, dtype=bool)

        output[name] = {"xy": xy, "visible": visible}
    return output


def load_validation_names(path):
    names = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.strip()
            if value:
                names.append(basename(value))
    if len(names) != len(set(names)):
        raise ValueError("validation list contains duplicate filenames")
    return names


def evaluate(gt, pred, validation_names, thresholds, expected_count):
    expected = set(validation_names)
    gt_names = set(gt)
    pred_names = set(pred)

    missing_gt = sorted(expected - gt_names)
    missing_pred = sorted(expected - pred_names)
    extra_pred = sorted(pred_names - expected)

    if missing_gt:
        raise ValueError("ground truth is missing {} validation images".format(len(missing_gt)))
    if missing_pred:
        raise ValueError("prediction is missing {} validation images".format(len(missing_pred)))
    if len(validation_names) != expected_count:
        raise ValueError(
            "validation list must contain {} images, got {}".format(
                expected_count, len(validation_names)
            )
        )

    all_errors = []
    per_image = []
    correct = {threshold: 0 for threshold in thresholds}
    total_valid = 0

    for name in validation_names:
        gt_xy = gt[name]["xy"]
        pred_xy = pred[name]["xy"]
        visible = gt[name]["visible"]
        errors = np.linalg.norm(pred_xy - gt_xy, axis=1)[visible]
        if errors.size == 0:
            continue

        all_errors.append(errors)
        total_valid += int(errors.size)
        for threshold in thresholds:
            correct[threshold] += int(np.count_nonzero(errors <= threshold))
        per_image.append(
            {
                "image": name,
                "valid_keypoints": int(errors.size),
                "mre_px": float(errors.mean()),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(errors.max()),
            }
        )

    if not all_errors:
        raise ValueError("no valid keypoints were evaluated")

    errors = np.concatenate(all_errors)
    metrics = {
        "coordinate_system": "original image pixels",
        "normalization": "none",
        "comparison_operator": "<=",
        "validation_images": len(validation_names),
        "valid_keypoints": total_valid,
        "mre_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "max_error_px": float(errors.max()),
        "pck_percent": {
            str(threshold): correct[threshold] / total_valid * 100.0
            for threshold in thresholds
        },
        "extra_prediction_images_ignored": extra_pred,
    }
    return metrics, per_image


def write_outputs(out_dir, metrics, per_image):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "per_image_errors.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    text_path = os.path.join(out_dir, "metrics.txt")
    lines = [
        "AASCE2019 pixel-coordinate evaluation",
        "Coordinate system: original image pixels",
        "Distance normalization: none",
        "Validation images: {}".format(metrics["validation_images"]),
        "Valid keypoints: {}".format(metrics["valid_keypoints"]),
        "MRE = {:.3f} px".format(metrics["mre_px"]),
    ]
    for threshold, value in metrics["pck_percent"].items():
        lines.append("PCK@{}px = {:.3f}%".format(threshold, value))
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return text_path, json_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate MRE and fixed-pixel PCK for AASCE 68 landmarks"
    )
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", default="pixel_metric_results")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[5, 10, 20])
    parser.add_argument("--num-keypoints", type=int, default=68)
    parser.add_argument("--expected-images", type=int, default=97)
    return parser.parse_args()


def main():
    args = parse_args()
    gt = records_from_json(args.gt_json, args.num_keypoints, is_ground_truth=True)
    pred = records_from_json(args.pred_json, args.num_keypoints, is_ground_truth=False)
    validation_names = load_validation_names(args.val_list)
    metrics, per_image = evaluate(
        gt,
        pred,
        validation_names,
        args.thresholds,
        args.expected_images,
    )
    text_path, json_path, csv_path = write_outputs(args.out_dir, metrics, per_image)

    print("AASCE2019 fixed-pixel evaluation")
    print("Validation images: {}".format(metrics["validation_images"]))
    print("Valid keypoints: {}".format(metrics["valid_keypoints"]))
    print("MRE = {:.3f} px".format(metrics["mre_px"]))
    for threshold, value in metrics["pck_percent"].items():
        print("PCK@{}px = {:.3f}%".format(threshold, value))
    print("Saved: {}, {}, {}".format(text_path, json_path, csv_path))


if __name__ == "__main__":
    main()
