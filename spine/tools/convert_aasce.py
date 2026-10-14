"""Prepare AASCE annotations using the original seed-42 conversion procedure.

Packaging changes: paths are CLI arguments; missing images, output collisions,
and a changed validation split are reported explicitly.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--val-list', type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError('Choose a new output directory: {}'.format(args.out_root))

    labels = args.raw_root / 'train_txt'
    filenames = pd.read_csv(labels / 'filenames.csv', header=None)[0].tolist()
    landmarks = pd.read_csv(labels / 'landmarks.csv', header=None).values
    angles = pd.read_csv(labels / 'angles.csv', header=None).values
    if len(set(filenames)) != len(filenames):
        raise ValueError('Duplicate dataset filenames')
    if landmarks.shape != (len(filenames), 136) or len(angles) != len(filenames):
        raise ValueError('CSV row counts or landmark dimensions do not match')

    records = []
    for i, name in enumerate(filenames):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected dataset-relative image filename')
        image = cv2.imread(str(args.raw_root / 'train' / relative), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError('Missing or unreadable image: {}'.format(name))
        height, width = image.shape[:2]
        row = landmarks[i].astype(np.float32)
        joints = np.stack([row[:68] * width, row[68:] * height], axis=1)
        records.append({
            'image': name, 'joints': joints.tolist(), 'joints_vis': [1] * 68,
            'center': [width / 2.0, height / 2.0],
            'scale': [width / 200.0, height / 200.0],
            'angles': angles[i].astype(float).tolist(),
        })

    random.Random(42).shuffle(records)
    split = int(len(records) * 0.8)
    train_records, val_records = records[:split], records[split:]
    expected = [line.strip() for line in args.val_list.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    actual = [record['image'] for record in val_records]
    if actual != expected:
        raise ValueError('Validation filenames/order differ from --val-list; check dataset version and CSV order')

    images_out = args.out_root / 'images'
    annotations_out = args.out_root / 'annot'
    images_out.mkdir(parents=True)
    annotations_out.mkdir()
    for name in filenames:
        target = images_out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.raw_root / 'train' / name, target)
    for name, rows in [('train', train_records), ('val', val_records)]:
        with (annotations_out / (name + '.json')).open('w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2)
    print('Prepared {} training and {} validation images in {}'.format(len(train_records), len(val_records), args.out_root))


if __name__ == '__main__':
    main()
