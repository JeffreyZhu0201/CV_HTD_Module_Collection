# CV_HTD_Module_Collection

本项目收集了多种计算机视觉与深度学习相关模块、经典模型复现代码，以及个人项目和学习笔记，涵盖了图像分割、目标检测、推荐系统等多个方向。

## 目录结构

```
CV_HTD_Module_Collection/
├── Github/                      # 经典/优秀的开源项目收集
│   ├── awesome-semantic-segmentation-pytorch/
│   ├── deep-learning-for-image-processing/
│   ├── Pixel-level-Hyperspectral-Target-Detection/
│   ├── TUFusion/
│   └── ZoRI-main/
├── MyGithub/                    # 个人项目与代码仓库
│   ├── Multimodal-DeepLearning-Based-Recommendation-System/
│   ├── pytorch-basic-framework/
│   ├── PytorchDeepLearningGuidance/
│   └── rainbow-pay-sdk-go/
├── PPT汇报/                     # 相关汇报PPT
├── 小项目/                      # 个人小型项目
├── 杂项/                        # 其他杂项内容
├── 经典模型复现/                # 经典模型复现代码
│   └── STARCOP/                 # 例如：STARCOP高光谱目标检测
└── programming-language-demo/   # 编程语言相关Demo
```

## 主要内容

- **经典模型复现**：如 [STARCOP](经典模型复现/STARCOP/README.md)，包含高光谱目标检测等前沿模型的复现与应用示例。
- **个人项目**：如 [rainbow-pay-sdk-go](MyGithub/rainbow-pay-sdk-go/)，实现了支付SDK的Go语言版本。
- **优秀开源项目收集**：如 [ZoRI-main](Github/ZoRI-main/INSTALL.md)，包含语义分割、目标检测等领域的主流项目。
- **学习笔记与PPT**：整理了学习过程中的笔记与相关汇报材料，便于查阅和复习。

## 环境配置

不同子项目有各自的环境需求，请参考各子目录下的 `README.md` 或 `requirements.txt` 文件。例如：

- Python 3.6/3.7/3.8/3.10
- Pytorch 1.10/2.x
- Ubuntu/Centos/Windows
- 其他依赖详见各项目说明

## 快速开始

以 STARCOP 为例：

```bash
conda create -c conda-forge -n starcop_env python=3.10 mamba
conda activate starcop_env
pip install git+https://github.com/spaceml-org/STARCOP.git
```

更多使用方法请参考各子项目的 `README.md` 文件。

## 贡献方式

欢迎提交 Issue 或 Pull Request，完善项目内容或修复相关问题。

## License

本项目遵循各子项目原始协议，详情请见各目录下 LICENSE 文件。