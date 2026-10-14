# Installation

## Environment

The scripts assume Linux, an NVIDIA GPU, and CUDA-enabled PyTorch. They call `.cuda()` directly; a CPU-only installation is not a supported execution path in this snapshot.

The inspected human-pose server environment contained Python 3.8.20, PyTorch 1.10.1+cu113, and torchvision 0.11.2+cu113. These are observed package versions, not proof of the environment for every historical experiment. Its installed NumPy was 1.24.3, whereas several retained loaders use `np.float`; the release requirements constrain NumPy below 1.24 to retain that legacy API.

A legacy environment starting point is:

```bash
conda create -n rdpose python=3.8 -y
conda activate rdpose
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install -r requirements.txt
```

The PyTorch pairing comes from the [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/#v1101). This environment recipe has not been installed or exercised as part of this source packaging. The requirements are compatibility constraints rather than a full experiment lockfile.

## Native NMS extensions

The human-pose dataset imports load the legacy CPU and GPU NMS extensions, even when a particular evaluation uses NumPy-based OKS NMS. Build them before importing the human-pose dataset package:

```bash
cd human_pose/lib/nms
python setup_linux.py build_ext --inplace
cd ../../..
```

This build needs a C/C++ compiler and a CUDA toolkit containing `nvcc`. The Conda CUDA runtime alone may not provide `nvcc`. The original build script looks for `CUDAHOME` or `nvcc` on `PATH` and targets `sm_35`; it is retained as legacy source. A newer toolkit that drops this target needs an appropriately updated architecture setting. Compiled `.so` files are local build outputs and must not be committed.

The three spinal dataset packages import MPII/AASCE directly and do not require these extensions for AASCE. If enabling their COCO/OCHuman imports, build the corresponding model directory's NMS extensions as well.

## Paths, GPUs, and weights

- Run scripts from `human_pose/`, `spine/hrnet/`, `spine/rdpose/`, or `spine/spine3/`, as shown in the task guides.
- Set `GPUS '(0,)'` for one visible GPU. Batch sizes in training configurations are per GPU; changing GPU count changes the effective total batch size.
- Use command-line overrides for `DATASET.ROOT`, `MODEL.PRETRAINED`, and `TEST.MODEL_FILE`. Absolute paths avoid ambiguity after reorganizing the repository.
- Obtain HRNet classification pretraining and ResNet weights from the upstream project. Trained RDPose checkpoints are not bundled.
- `tools/test.py` expects a model state dictionary compatible with the selected architecture. Prefer the saved `model_best.pth` or `final_state.pth` over a resume checkpoint containing optimizer state.
- The AASCE prediction exporter also accepts a dictionary containing `state_dict` and strips a leading `module.` prefix.

The original human-pose dependency list is retained as `human_pose/requirements.original.txt` for provenance. Use the root requirements for the documented release environment.
