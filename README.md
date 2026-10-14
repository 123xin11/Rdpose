# RDPose

Research code for **RDPose: Robust Human Pose Estimation via Feature Decoupling and Perturbation Learning**, with an AASCE2019 spinal landmark adaptation.

**Authors:** Xin Liu, Qi Zou, Zhiyuan Liu, and Ziyang Lyu.

[中文说明](README.zh-CN.md) · [Installation](docs/installation.md) · [Human pose](docs/human_pose.md) · [Spinal landmarks](docs/spine.md) · [Release notes](docs/release_notes.md)

## Repository layout

| Directory | Contents |
| --- | --- |
| `human_pose/` | RDPose with HRNet and ResNet backbones; COCO/MPII configurations and an OCHuman loader |
| `spine/hrnet/` | HRNet baseline for 68 spinal landmarks |
| `spine/rdpose/` | RDPose adaptation for 68 spinal landmarks |
| `spine/spine3/` | Three-segment center supervision and residual feature refinement, including residual-scale configurations |
| `spine/tools/` | Dataset preparation and pixel-coordinate evaluation utilities |
| `splits/aasce2019/` | The 97-image custom validation list, containing dataset filenames only |
| `docs/` | Installation, data formats, commands, and source provenance |
| `licenses/` | Upstream license texts |

Each model directory retains its own `lib/`, `tools/`, and `experiments/` tree. Run commands from the model directory indicated in the documentation: these trees use different training interfaces and should not be combined through a shared `PYTHONPATH`.

## Getting started

```bash
git clone https://github.com/123xin11/Rdpose.git
cd Rdpose
```

1. Create the environment described in [Installation](docs/installation.md).
2. Obtain datasets and pretrained backbone checkpoints from their original providers.
3. Follow the commands in [Human pose](docs/human_pose.md) or [Spinal landmarks](docs/spine.md).

The scripts use PyTorch and CUDA. Training, validation, and prediction export entry points are included. Dataset images, annotation coordinates, person detections, trained model weights, and generated experiment outputs are not distributed in this source release.

## Release scope

This release organizes the available research source snapshots. The original model and loss implementations are retained. Configuration defaults and command-line overrides are documented explicitly, including person-box sources and spinal postprocessing. Full training and inference were not rerun for this packaging release, and this commit is not a claim that every manuscript result has been independently reproduced.

The three-segment spinal model and its additional configurations are an adaptation of the human-pose implementation. See [release notes](docs/release_notes.md) for source mapping and reproducibility boundaries.

## Acknowledgments and licensing

The code builds on [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation), [Simple Baselines](https://github.com/microsoft/human-pose-estimation.pytorch), and their dependencies. Original copyright headers are preserved. See [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component-specific terms.

For questions about the released implementation, please use [GitHub Issues](https://github.com/123xin11/Rdpose/issues).
