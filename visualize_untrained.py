import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import os


# --- 1. [必需] 粘贴 Gabor 类的定义 ---
# (来自您上传的 Gradient_model3.py 文件)
class PhaseEnhancedGaborFilter(nn.Module):
    """
    自适应 Gabor 滤波器 + 多尺度频率 + 高频相位增强
    用于红外小目标检测：增强对高频纹理与边缘的感知
    """

    def __init__(self, in_channels, kernel_size, orientations=8, n_freqs=1, init_phase_weight=2.7, theta=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.orientations = orientations
        self.n_freqs = n_freqs  # 每个方向多个频率

        # 相位增强权重 - 使用对数参数化确保正权重
        self.log_phase_weight = nn.Parameter(
            torch.tensor(math.log(init_phase_weight), dtype=torch.float32),
            requires_grad=True
        )
        # self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32), requires_grad=True)

        # 可学习参数（频率、sigma、gamma）
        self.log_lambd = nn.Parameter(torch.zeros(orientations, n_freqs))  # log λ
        self.log_sigma = nn.Parameter(torch.zeros(orientations, n_freqs))  # log σ
        self.gamma = nn.Parameter(torch.ones(orientations, n_freqs) * 0.5)  # γ

        self._initialize_parameters()

    def _initialize_parameters(self):
        # 仅在 kernel_size 大于 0 时初始化
        if self.kernel_size <= 0:
            print(f"警告: kernel_size 为 {self.kernel_size}，跳过参数初始化。")
            return

        init_lambd = self.kernel_size * 0.5
        init_sigma = 0.56 * init_lambd

        print(f"  > [Gabor Init] k_size={self.kernel_size}, init_lambd={init_lambd:.2f}, init_sigma={init_sigma:.2f}")

        for i in range(self.orientations):
            for j in range(self.n_freqs):
                scale = 1.0 / (2 ** j)  # 高频 → 更小的 λ
                self.log_lambd.data[i, j] = math.log(init_lambd * scale)
                self.log_sigma.data[i, j] = math.log(init_sigma * scale)
                self.gamma.data[i, j] = 0.5

    @property
    def phase_weight(self):
        """获取经过约束的相位增强权重"""
        # 使用 softplus 确保权重始终为正
        return F.softplus(self.log_phase_weight)

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        center = self.kernel_size // 2

        # 坐标网格
        coords = torch.arange(-center, center + 1, dtype=torch.float32, device=device)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

        outputs = []
        for i in range(self.orientations):
            theta = i * math.pi / self.orientations
            cos_theta = torch.cos(torch.tensor(theta, device=device))
            sin_theta = torch.sin(torch.tensor(theta, device=device))

            # 坐标旋转
            x_theta = x_grid * cos_theta + y_grid * sin_theta
            y_theta = -x_grid * sin_theta + y_grid * cos_theta

            for j in range(self.n_freqs):
                lambd = torch.exp(self.log_lambd[i, j])
                sigma = torch.exp(self.log_sigma[i, j])
                gamma = torch.clamp(self.gamma[i, j], 0.1, 1.0)

                # ====== 复数 Gabor ======
                exponent = -(x_theta ** 2 + (gamma ** 2) * (y_theta ** 2)) / (2 * sigma ** 2)
                gauss = torch.exp(exponent)

                real = gauss * torch.cos(2 * math.pi * x_theta / lambd)
                imag = gauss * torch.sin(2 * math.pi * x_theta / lambd)
                complex_kernel = torch.complex(real, imag)

                # ====== 高频相位增强优化 ======
                phase_grad_magnitude = (2 * math.pi / lambd) * torch.abs(cos_theta)

                enhanced_magnitude = torch.abs(complex_kernel) * (
                        1.0 + self.phase_weight * phase_grad_magnitude
                )

                # 3. 带通归一化
                enhanced_magnitude = enhanced_magnitude - enhanced_magnitude.mean()
                norm_factor = torch.sum(torch.abs(enhanced_magnitude))
                if norm_factor > 1e-8:
                    enhanced_magnitude = enhanced_magnitude / norm_factor

                # 卷积
                kernel = enhanced_magnitude.view(1, 1, self.kernel_size, self.kernel_size)
                expanded_kernel = kernel.repeat(c, 1, 1, 1)
                padding = self.kernel_size // 2
                output = F.conv2d(x, expanded_kernel, padding=padding, groups=c)
                outputs.append(output)

        return torch.cat(outputs, dim=1)  # [b, c*orientations*n_freqs, h, w]


# --- 2. 归一化函数 ---
def normalize_map(data):
    """将特征图数据归一化到 0-1 范围以便显示"""
    data = data.astype(np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val > 1e-6:
        return (data - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(data) if min_val == 0 else np.ones_like(data)


# --- 3. 主可视化函数 ---
def visualize_untrained_gabor_outputs():
    CAPTURE_DIR = 'gabor_feature_captures'
    VIS_OUTPUT_DIR = 'gabor_untrained_visualization'
    ORIENTATIONS = 8

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CAPTURE_DIR):
        print(f"错误: 目录 '{CAPTURE_DIR}' 未找到。")
        print("请确保此脚本与 'gabor_feature_captures' 文件夹在同一目录中。")
        return

    print(f"正在扫描目录: {CAPTURE_DIR}")

    all_files = os.listdir(CAPTURE_DIR)
    input_files = [f for f in all_files if f.endswith('_input.pt')]

    if not input_files:
        print(f"在 '{CAPTURE_DIR}' 目录中未找到 '_input.pt' 文件。")
        return

    print(f"找到了 {len(input_files)} 个 Gabor 输入文件。正在生成未训练的输出...")
    vis_count = 0

    for input_file in input_files:
        base_name = input_file.replace('_input.pt', '')
        input_path = os.path.join(CAPTURE_DIR, input_file)

        print(f"\n正在处理: {base_name}")

        try:
            # --- 加载输入数据 ---
            x_tensor = torch.load(input_path, map_location='cpu')  # [1, C_in, H, W]

            # --- 推断模型参数 ---
            # 从 ExpansionContrastModule 定义 [5, 7] 推断
            if 'gabor_filters.0' in base_name:
                kernel_size = 5
            elif 'gabor_filters.1' in base_name:
                kernel_size = 7
            else:
                print(f"  - [警告] 无法从文件名 '{base_name}' 推断 kernel_size。跳过。")
                continue

            in_channels = x_tensor.shape[1]
            print(f"  - 推断参数: in_channels={in_channels}, kernel_size={kernel_size}")

            # --- 实例化 *未训练* 的模型 ---
            model_untrained = PhaseEnhancedGaborFilter(
                in_channels=in_channels,
                kernel_size=kernel_size,
                orientations=ORIENTATIONS
            )
            model_untrained.eval()  # 设置为评估模式

            # --- 运行前向传播 ---
            with torch.no_grad():
                output_untrained = model_untrained(x_tensor)  # [1, C_in*8, H, W]

            # --- 准备可视化数据 ---
            # 1. 聚合输入 (在 C_in 个通道上求和)
            x_vis = x_tensor.squeeze(0).sum(dim=0).numpy()

            # 2. 聚合输出 (拆分为 8 个方向，每个方向再求和)
            out_tensor_squeezed = output_untrained.squeeze(0)  # [C_in*8, H, W]
            output_chunks = torch.chunk(out_tensor_squeezed, chunks=ORIENTATIONS, dim=0)

            output_vis_list = []
            for i in range(ORIENTATIONS):
                chunk_vis = output_chunks[i].sum(dim=0).numpy()
                output_vis_list.append(chunk_vis)

            # --- 绘图 ---
            fig, axes = plt.subplots(1, 1 + ORIENTATIONS, figsize=(20, 3))

            axes[0].imshow(normalize_map(x_vis), cmap='gray')
            axes[0].set_title("Input")
            axes[0].axis('off')

            for i in range(ORIENTATIONS):
                ax = axes[i + 1]
                ax.imshow(normalize_map(output_vis_list[i]), cmap='gray')
                ax.set_title(f'Untrained θ={i*22.5: .1f}°')
                ax.axis('off')

            fig.suptitle(f'Untrained Output for: {base_name}', y=1.05, fontsize=14)
            plt.tight_layout()

            # 保存图像
            save_name = f"VIS_UNTRAINED_{base_name}.png"
            save_path = os.path.join(VIS_OUTPUT_DIR, save_name)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            print(f"  -> 已保存可视化图像: {save_path}")
            vis_count += 1

        except Exception as e:
            print(f"  - [错误] 处理 {base_name} 时失败: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误

    print(f"\n全部完成。共生成 {vis_count} 张未训练的可视化图像，保存在 '{VIS_OUTPUT_DIR}' 目录中。")


# --- 4. 运行脚本 ---
if __name__ == "__main__":
    visualize_untrained_gabor_outputs()