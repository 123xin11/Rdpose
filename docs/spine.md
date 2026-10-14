# AASCE2019 spinal landmark adaptation

## Implementations

| Working directory | Main configuration under `experiments/aasce/hrnet/` |
| --- | --- |
| `spine/hrnet/` | `w32_256x512_fullcrop_adam_lr1e-3.yaml` |
| `spine/rdpose/` | `w32_256x512_fullcrop_adam_lr1e-3.yaml` |
| `spine/spine3/` | `w32_256x512_spine3_fullcrop_adam_lr1e-3.yaml` |
| `spine/spine3/` | `w32_256x512_spine3_fdm005_fullcrop_adam_lr1e-3.yaml` |
| `spine/spine3/` | `w32_256x512_spine3_fdm010_fullcrop_adam_lr1e-3.yaml` |
| `spine/spine3/` | `w32_256x512_spine3_fdm020_fullcrop_adam_lr1e-3.yaml` |

Additional auxiliary/endpoint configuration experiments from the source snapshot are retained. Their inclusion does not mean they are the configurations used for every manuscript table.

## Prepare data

Obtain AASCE2019 images and labels separately. The loaders use a list of records in `annot/train.json` and `annot/val.json`. Each record has:

- `image`: filename relative to the dataset's `images/` directory;
- `joints`: 68 pairs of **original-image pixel coordinates**;
- `joints_vis`: 68 visibility indicators;
- `center`: crop center in original-image pixels;
- `scale`: crop width and height divided by 200.

The supplied validation list, `splits/aasce2019/val_fullcrop_97.txt`, was extracted from saved prediction records. HRNet, RDPose, and Spine3-FDM-0.10 records contain the same 97 filenames in the same order. It is a custom split, separate from the official 128-image validation set. No annotation coordinates are distributed.

The portable conversion utility follows the original seed-42, 80/20 preparation script and checks its resulting validation order against this list:

```bash
# Run from the repository root; output directory must be new.
python spine/tools/convert_aasce.py \
  --raw-root /path/to/AASCE2019 \
  --out-root /path/to/aasce_prepared \
  --val-list splits/aasce2019/val_fullcrop_97.txt
```

It expects `train/` images and `train_txt/filenames.csv`, `landmarks.csv`, and `angles.csv`. Landmark CSV columns contain 68 normalized x values followed by 68 normalized y values. Conversion fails if required images are missing or the split differs, rather than silently changing the sample list.

For the fullcrop condition, run the retained crop-preparation script:

```bash
python spine/spine3/tools/regenerate_aasce_center_scale.py \
  --root /path/to/aasce_prepared --suffix _fullcrop \
  --image-size 256,512 --padding 1.25
```

This writes `annot/train_fullcrop.json` and `annot/val_fullcrop.json`. It derives the crop from the visible ground-truth landmark extent, adjusts it to the target aspect ratio, and adds padding. This is an annotation-assisted crop protocol. The YAML filename alone does not select these new annotation files; pass the split names explicitly as below.

## Train Spine3-FDM-0.10

```bash
cd spine/spine3
python tools/train.py \
  --cfg experiments/aasce/hrnet/w32_256x512_spine3_fdm010_fullcrop_adam_lr1e-3.yaml \
  DATASET.ROOT /path/to/aasce_prepared \
  DATASET.TRAIN_SET train_fullcrop DATASET.TEST_SET val_fullcrop \
  MODEL.PRETRAINED /path/to/hrnet_w32-36af842e.pth \
  OUTPUT_DIR output LOG_DIR log GPUS '(0,)' \
  TEST.SPINE_CURVE_POST_PROCESS False
```

The input is 256 pixels wide by 512 high; the heatmaps are 64 by 128. The configurations use 68 anatomical landmarks. Auxiliary center supervision is handled within the model/dataset implementation.

For HRNet or original RDPose, change to the appropriate working directory and select its fullcrop YAML. Those configurations do not share every Spine3-specific setting: omit `TEST.SPINE_CURVE_POST_PROCESS` when it is not declared by that model's configuration module.

## Export predictions

```bash
python tools/save_aasce_preds.py \
  --cfg experiments/aasce/hrnet/w32_256x512_spine3_fdm010_fullcrop_adam_lr1e-3.yaml \
  --model-file /path/to/spine3_model_best.pth \
  --outdir predictions/spine3_fdm010 \
  DATASET.ROOT /path/to/aasce_prepared DATASET.TEST_SET val_fullcrop \
  GPUS '(0,)' TEST.SPINE_CURVE_POST_PROCESS False
```

The exporter saves `all_preds.npy`, `all_boxes.npy`, and `image_paths.txt`. Coordinates are transformed back to the original image frame. Only the first 68 anatomical landmark channels are exported. This exporter does not perform flip testing.

**Postprocessing:** the preserved alpha-0.10 YAML enables `TEST.SPINE_CURVE_POST_PROCESS`, while the other residual-scale configurations do not all enable it. The example explicitly disables it for a common decoding setting. For any comparison, record the residual scale, quarter-pixel setting, curve-correction flag, split, and checkpoint. The example is a documented evaluation setup, not a claim that disabling the flag reproduces a particular manuscript score.

## Metrics

The AASCE dataset evaluator reports MRE and PCK at 5, 10, and 20 pixels. `compare_spine3_fdm_ablation.py` analyzes saved arrays using these pixel-error metrics.

The additional `spine/tools/calc_aasce_pixel_metrics.py` evaluates ground-truth and prediction JSON files by image filename. Each JSON record should contain `image` and `keypoints` (68 pairs or 68 triples); ground truth may also include `joints_vis`. For this utility, rename an annotation record's `joints` field to `keypoints` when preparing its evaluation input.

```bash
# Run from the repository root.
python spine/tools/calc_aasce_pixel_metrics.py \
  --gt-json /path/to/gt_pixel_keypoints.json \
  --pred-json /path/to/pred_pixel_keypoints.json \
  --val-list splits/aasce2019/val_fullcrop_97.txt \
  --thresholds 5 10 20 40 --expected-images 97 \
  --out-dir outputs/pixel_metrics
```

For the same valid landmark set, OR20 = 100 - PCK@20px and OR40 = 100 - PCK@40px. These utilities do not report NMSE. Computing NMSE requires the explicitly chosen normalization definition; it must not be inferred from PCK or from the heatmap training loss.
