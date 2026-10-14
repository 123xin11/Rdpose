# RDPose 代码仓库

本仓库整理了 RDPose 人体姿态估计代码，以及基于 AASCE2019 的脊柱关键点定位扩展。

论文题名：**RDPose: Robust Human Pose Estimation via Feature Decoupling and Perturbation Learning**。

作者：Xin Liu、Qi Zou、Zhiyuan Liu、Ziyang Lyu。

## 内容

- `human_pose/`：人体姿态模型、训练与评估脚本、COCO/MPII 配置及 OCHuman 数据读取代码。
- `spine/hrnet/`：68 个脊柱关键点的 HRNet 基线。
- `spine/rdpose/`：RDPose 脊柱适配版本。
- `spine/spine3/`：三段中心监督、残差特征增强及残差系数配置。
- `spine/tools/`：数据转换与像素坐标评估工具。
- `splits/aasce2019/`：自定义 97 张验证图像的文件名清单。

安装步骤见 [Installation](docs/installation.md)，运行方法见 [人体姿态说明](docs/human_pose.md)和[脊柱实验说明](docs/spine.md)。命令应在文档指定的模型目录内执行，各模型的 `lib` 不应混用。

## 使用说明

原始数据、标注坐标、检测框文件、预训练权重和训练后的权重需要另外准备。本次发布包含代码和配置，没有上传医学图像、运行日志或模型权重。

本次整理保留原有模型与损失实现，补充了路径设置和运行说明，未重新执行完整训练或推理。原始 COCO 配置中的框来源、MPII 的测试集选项、脊柱曲线后处理等会影响结果，文档中已列明。请用同一数据划分、框来源和后处理设置进行对比。

版本来源和整理范围见 [Release notes](docs/release_notes.md)。第三方代码保留原有版权声明；具体许可见 [LICENSE.md](LICENSE.md)。
