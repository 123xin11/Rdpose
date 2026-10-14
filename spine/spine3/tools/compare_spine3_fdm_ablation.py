import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import _init_paths  # noqa

from config import cfg, update_config
import dataset
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Compare Spine3-Aux and Spine3-FDM ablation results')
    parser.add_argument('--cfg', required=True, type=str)
    parser.add_argument('--pred-spine3-aux', required=True, type=str)
    parser.add_argument('--pred-fdm005', default='', type=str)
    parser.add_argument('--pred-fdm010', default='', type=str)
    parser.add_argument('--pred-fdm020', default='', type=str)
    parser.add_argument('--outdir', default='spine3_fdm_ablation_analysis', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def compute_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        'MRE': float(np.mean(errors)),
        'Median': float(np.median(errors)),
        'Max': float(np.max(errors)),
        'PCK5': float(np.mean(errors <= 5) * 100.0),
        'PCK10': float(np.mean(errors <= 10) * 100.0),
        'PCK20': float(np.mean(errors <= 20) * 100.0),
    }


def main():
    args = parse_args()
    update_config(cfg, args)
    os.makedirs(args.outdir, exist_ok=True)

    val_dataset = dataset.aasce(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.ToTensor()
    )

    pred_files = {'Spine3-Aux': args.pred_spine3_aux}
    if args.pred_fdm005:
        pred_files['Spine3-FDM-0.05'] = args.pred_fdm005
    if args.pred_fdm010:
        pred_files['Spine3-FDM-0.10'] = args.pred_fdm010
    if args.pred_fdm020:
        pred_files['Spine3-FDM-0.20'] = args.pred_fdm020

    preds = {name: np.load(path) for name, path in pred_files.items()}
    overall_rows = []
    point_rows = []

    for name, arr in preds.items():
        all_errors = []
        for idx, rec in enumerate(val_dataset.db):
            gt = rec['joints_3d'][:68, 0:2]
            vis = rec['joints_3d_vis'][:68, 0] > 0
            pred = arr[idx, :68, 0:2]
            err = np.linalg.norm(pred - gt, axis=1)
            for j in range(68):
                if not vis[j]:
                    continue
                all_errors.append(float(err[j]))
                point_rows.append({
                    'method': name,
                    'index': idx,
                    'image': os.path.basename(rec['image']),
                    'landmark_id': j,
                    'vertebra_id': j // 4,
                    'corner_id': j % 4,
                    'error': float(err[j]),
                    'gt_x': float(gt[j, 0]),
                    'gt_y': float(gt[j, 1]),
                    'pred_x': float(pred[j, 0]),
                    'pred_y': float(pred[j, 1]),
                })
        m = compute_metrics(all_errors)
        m['method'] = name
        overall_rows.append(m)

    overall_df = pd.DataFrame(overall_rows)[['method', 'MRE', 'Median', 'Max', 'PCK5', 'PCK10', 'PCK20']]
    point_df = pd.DataFrame(point_rows)

    vertebra_rows = []
    for name in preds.keys():
        sub = point_df[point_df['method'] == name]
        per_v = sub.groupby('vertebra_id')['error'].agg(
            mean='mean', median='median', max='max', count='count'
        ).reset_index()
        per_v['pck20'] = sub.groupby('vertebra_id')['error'].apply(lambda x: np.mean(x <= 20) * 100.0).values
        per_v['method'] = name
        vertebra_rows.append(per_v)

    vertebra_df = pd.concat(vertebra_rows, ignore_index=True)
    pivot_mean = vertebra_df.pivot(index='vertebra_id', columns='method', values='mean').reset_index()
    pivot_pck20 = vertebra_df.pivot(index='vertebra_id', columns='method', values='pck20').reset_index()

    overall_df.to_csv(os.path.join(args.outdir, 'overall_ablation_summary.csv'), index=False)
    point_df.to_csv(os.path.join(args.outdir, 'per_point_ablation_errors.csv'), index=False)
    vertebra_df.to_csv(os.path.join(args.outdir, 'per_vertebra_ablation_summary.csv'), index=False)
    pivot_mean.to_csv(os.path.join(args.outdir, 'per_vertebra_mean_compare.csv'), index=False)
    pivot_pck20.to_csv(os.path.join(args.outdir, 'per_vertebra_pck20_compare.csv'), index=False)

    print('\n===== Overall ablation summary =====')
    print(overall_df.to_string(index=False))
    print('\n===== Per-vertebra mean compare =====')
    print(pivot_mean.to_string(index=False))
    print('\nSaved to:', args.outdir)


if __name__ == '__main__':
    main()
