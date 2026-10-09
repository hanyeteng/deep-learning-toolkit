# Deep Learning Toolkit

用于深度学习实验的配套工具代码，涵盖模型评估、结果可视化和实验分析，当前示例主要面向红外小目标检测。

## 工具说明

| 脚本 | 功能 |
| --- | --- |
| `get_ROC.py` | 绘制多模型 ROC 曲线并计算 AUC |
| `params_compute.py` | 统计参数量、计算量、推理时延、FPS 和 GPU 显存占用 |
| `plot_efficient.py` | 绘制参数量、IoU 与 FLOPs 的效率对比气泡图 |
| `feature_map_visualization.py` | 提取并保存模型中间特征图及预测结果 |
| `noised_dataset.py` | 为图像添加高斯噪声和散斑噪声 |
| `visualization.py` | 可视化已保存的 Gabor 特征及滤波响应 |
| `visualize_untrained.py` | 可视化未训练 Gabor 滤波器的响应 |

## 使用方法

克隆仓库：

```bash
git clone https://github.com/hanyeteng/deep-learning-toolkit.git
cd deep-learning-toolkit
```

常用依赖包括 PyTorch、torchvision、NumPy、Matplotlib、OpenCV、scikit-learn、pandas、tqdm 和 THOP，请按所使用的脚本及本机 CUDA 环境安装。

绘制效率对比图：

```bash
python plot_efficient.py
```

运行前修改脚本中的 `methods`、`params`、`flops` 和 `iou` 数据。支持通过 `--show` 显示图表，通过 `--output-dir` 指定输出目录，图表导出为 PDF、PNG 和 SVG。

## 使用说明

- 部分脚本依赖原项目中的 `net.py`、`utils/` 和 `evaluation/`，需要补充对应模块或集成到原项目中使用。
- 数据集和模型权重不包含在本仓库中，请根据实际项目配置路径与模型名称。
- `params_compute.py` 当前使用 CUDA；噪声生成脚本的输入、输出路径需自行修改。

## 许可证

本项目采用 [MIT License](LICENSE)。
