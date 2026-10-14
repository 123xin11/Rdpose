# Human pose estimation

## Data and checkpoints

Run the following examples from `human_pose/`. Obtain COCO, MPII, OCHuman, their annotations, and any person detections separately. Follow the [upstream HRNet preparation instructions](https://github.com/HRNet/HRNet-Human-Pose-Estimation#data-preparation) for COCO/MPII conversion and backbone pretraining.

Expected directory structures are:

```text
data/coco/
  annotations/person_keypoints_train2017.json
  annotations/person_keypoints_val2017.json
  images/train2017/*.jpg
  images/val2017/*.jpg
  person_detection_results/COCO_val2017_detections_AP_H_56_person.json
data/mpii/
  annot/train.json
  annot/val.json
  annot/gt_val.mat
  images/...
data/ochuman/
  ochuman_coco_format_val_range_0.00_1.00.json
  ochuman_coco_format_test_range_0.00_1.00.json
  images/...
```

For OCHuman, the loader expects the COCO-format annotations named above. Native annotations require conversion before use; a conversion script is not included in this snapshot.

## COCO / HRNet-W32

```bash
cd human_pose
python tools/train.py \
  --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
  DATASET.ROOT data/coco \
  MODEL.PRETRAINED models/hrnet_w32-36af842e.pth \
  GPUS '(0,)' WORKERS 4 DEBUG.DEBUG False \
  TEST.USE_GT_BBOX False \
  TEST.COCO_BBOX_FILE data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json
```

The preserved W32 YAML has an empty `DATASET.ROOT` and `TEST.USE_GT_BBOX: true`. The example sets both explicitly to use detector boxes during validation. To evaluate a trained checkpoint:

```bash
python tools/test.py \
  --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
  DATASET.ROOT data/coco \
  TEST.MODEL_FILE /path/to/rdpose_w32_model_best.pth \
  GPUS '(0,)' WORKERS 4 DEBUG.DEBUG False \
  TEST.USE_GT_BBOX False \
  TEST.COCO_BBOX_FILE data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json
```

W48, other input sizes, and ResNet configurations are in `experiments/coco/`. Use matching backbone weights and checkpoint architectures. YAML image and heatmap sizes are stored as **[width, height]**, although some filenames use the height-by-width convention.

Keep the split, box source, detection file, flip testing, heatmap shift, and quarter-pixel postprocessing identical when comparing methods. Configurations include flipping and quarter-pixel decoding; consult the YAML and the merged configuration logged by the script for the exact settings. Changing to test-dev also requires its images, annotation metadata, detection file, and an evaluation-server submission; the example above evaluates `val2017` only.

## MPII validation

The original MPII YAML selects `TEST_SET: test`. For validation-set PCKh, override it to `val` and provide `annot/gt_val.mat` (the loader constructs `gt_<split>.mat`):

```bash
python tools/train.py \
  --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
  DATASET.ROOT data/mpii DATASET.TEST_SET val \
  MODEL.PRETRAINED models/hrnet_w32-36af842e.pth \
  GPUS '(0,)' WORKERS 4 DEBUG.DEBUG False

python tools/test.py \
  --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
  DATASET.ROOT data/mpii DATASET.TEST_SET val \
  TEST.MODEL_FILE /path/to/rdpose_mpii_model_best.pth \
  GPUS '(0,)' WORKERS 4 DEBUG.DEBUG False
```

## OCHuman evaluation

The loader is included, but there is no dedicated OCHuman YAML in the source snapshot. Use the COCO architecture configuration and explicitly select the dataset, its COCO-format annotations, and the box protocol. For example, the following uses **ground-truth person boxes**:

```bash
python tools/test.py \
  --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
  DATASET.DATASET ochuman DATASET.ROOT data/ochuman DATASET.TEST_SET val \
  TEST.MODEL_FILE /path/to/coco_trained_rdpose_w32.pth \
  TEST.USE_GT_BBOX True GPUS '(0,)' WORKERS 4 DEBUG.DEBUG False
```

For a detector-box protocol, use `TEST.USE_GT_BBOX False` and supply a compatible detection JSON through `TEST.COCO_BBOX_FILE`. GT-box and detector-box scores are different experimental settings.

## Outputs

Training creates model checkpoints and logs under the configured output/log directories. The scripts copy the model source into the experiment directory and log merged configuration values. Preserve the config, checkpoint, dataset version, and box file together for each run. The human-pose validation loop also writes a temporary `images_grid.jpg`, which is ignored by Git.
