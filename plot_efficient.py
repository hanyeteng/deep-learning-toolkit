"""Parameter / IoU / FLOPs comparison on IRSTD-1K."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

# 数据配置 (IRSTD-1K)，保持表格中的数值和顺序。
methods = [
    'ACM', 'ALCNet', 'AGPCNet', 'DNANet', 'UIUNet', 'RDIAN', 'ISTDU-Net',
    'SCTransNet', 'DATransNet', 'GSFANet', 'FM-Net', 'MDAFNet', 'DCCS-Det',
    'ADGFNet-Lite', 'ADGFNet', 'FSGPNet'
]
params = [
    0.40, 0.38, 12.43, 4.70, 50.54, 0.22, 2.75,
    11.19, 2.18, 6.12, 3.61, 4.45, 7.09, 0.12, 0.58, 4.75
]
flops = [
    1.61, 14.97, 173.02, 57.04, 217.70, 14.87, 31.78,
    40.46, 32.73, 20.46, 14.17, 27.24, 31.86, 3.58, 31.53, 50.02
]
iou = [
    63.87, 64.94, 67.73, 65.73, 65.69, 59.93, 65.01,
    66.58, 65.69, 65.23, 66.99, 64.77, 67.03, 66.45, 66.70, 68.44
]

DATASET_NAME = 'IRSTD-1K'
# 保留原脚本排除 ACM 的选择；设为空集合即可显示全部方法。
EXCLUDED_METHODS = {'ACM'}
PROPOSED_METHODS = {'ADGFNet-Lite', 'ADGFNet', 'FSGPNet'}
BASELINE_COLOR = '#5A8FB4'
PROPOSED_COLOR = '#C3413B'
OUTPUT_STEM = 'Efficiency_Bubble_Plot'
# 标签偏移使用 points，避免气泡尺寸与数据坐标混用。
LABEL_OFFSETS = {
    'ACM': (10, -16), 'ALCNet': (-4, -24), 'AGPCNet': (10, 23),
    'DNANet': (18, 3), 'UIUNet': (-8, -32), 'RDIAN': (10, 12),
    'ISTDU-Net': (-18, -28), 'SCTransNet': (14, -22),
    'DATransNet': (-16, 22), 'GSFANet': (18, -22), 'FM-Net': (-16, 23),
    'MDAFNet': (-4, -48), 'DCCS-Det': (-10, 18),
    'ADGFNet-Lite': (12, -20), 'ADGFNet': (8, 18), 'FSGPNet': (4, 22),
}


def bubble_size(values):
    """Marker area is proportional to FLOPs; legends use the same mapping."""
    return np.asarray(values, dtype=float) * 5.0


def build_figure():
    """Build the plot without saving files or opening a GUI."""
    if not (len(methods) == len(params) == len(flops) == len(iou)):
        raise ValueError('methods, params, flops and iou must have equal lengths.')
    values = np.column_stack((params, flops, iou)).astype(float)
    if not np.isfinite(values).all() or (values[:, :2] <= 0).any():
        raise ValueError('Values must be finite; parameters and FLOPs must be positive.')
    keep = np.array([name not in EXCLUDED_METHODS for name in methods])
    names = [name for name, included in zip(methods, keep) if included]
    if not names:
        raise ValueError('At least one method must be included.')
    parameters, operations, scores = values[keep].T
    proposed = np.array([name in PROPOSED_METHODS for name in names])
    sizes = bubble_size(operations)
    style = {
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11, 'axes.labelsize': 13, 'axes.labelweight': 'semibold',
        'axes.linewidth': 0.9, 'axes.edgecolor': '#747B83',
        'text.color': '#27313D', 'axes.labelcolor': '#27313D',
        'xtick.color': '#46515D', 'ytick.color': '#46515D',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=160)
        fig.subplots_adjust(left=0.09, right=0.975, bottom=0.26, top=0.91)
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.set_xscale('log')
        ax.set_xlim(parameters.min() / 1.45, parameters.max() * 1.65)
        score_span = max(float(np.ptp(scores)), 1.0)
        ax.set_ylim(scores.min() - score_span * 0.13,
                    scores.max() + score_span * 0.16)
        ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='both', which='major', length=4, width=0.8, pad=6)
        ax.tick_params(axis='x', which='minor', length=2, color='#BAC0C7')
        ax.grid(axis='y', color='#DCE1E7', linestyle='--', linewidth=0.7)
        ax.grid(axis='x', which='major', color='#E7EBEF', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Parameters (M, logarithmic scale)', labelpad=11)
        ax.set_ylabel('IoU (%)', labelpad=10)
        # 基线先绘制，本文方法置于上层。
        for selection, color, alpha, zorder in (
            (~proposed, BASELINE_COLOR, 0.65, 3),
            (proposed, PROPOSED_COLOR, 0.9, 4),
        ):
            ax.scatter(parameters[selection], scores[selection],
                       s=sizes[selection], c=color, alpha=alpha,
                       edgecolors='white', linewidths=1.1, zorder=zorder)
        for name, x, y, highlighted in zip(names, parameters, scores, proposed):
            dx, dy = LABEL_OFFSETS.get(name, (10, 16))
            ax.annotate(
                name, xy=(x, y), xytext=(dx, dy), textcoords='offset points',
                ha='left' if dx >= 0 else 'right',
                va='bottom' if dy >= 0 else 'top',
                fontsize=11.5 if highlighted else 10.5,
                fontweight='bold' if highlighted else 'normal',
                color=PROPOSED_COLOR if highlighted else '#27313D',
                bbox={'boxstyle': 'round,pad=0.12', 'fc': 'white',
                      'ec': 'none', 'alpha': 0.9},
                arrowprops={'arrowstyle': '-',
                            'color': PROPOSED_COLOR if highlighted else '#9AA6B2',
                            'lw': 0.65, 'shrinkA': 2, 'shrinkB': 4,
                            'connectionstyle': 'arc3'}, zorder=6,
            )
        fig.text(0.09, 0.952, DATASET_NAME, fontsize=15, fontweight='bold')
        fig.text(0.975, 0.952, 'Performance and model efficiency',
                 ha='right', fontsize=11, color='#64707D')
        method_handles = [
            Line2D([], [], marker='o', linestyle='none', markersize=9,
                   markerfacecolor=color, markeredgecolor='white', label=label)
            for color, label in [(BASELINE_COLOR, 'Baseline methods'),
                                 (PROPOSED_COLOR, 'Proposed methods')]
        ]
        fig.legend(handles=method_handles, loc='center left',
                   bbox_to_anchor=(0.09, 0.105), frameon=False,
                   fontsize=10.5, labelspacing=0.85, handletextpad=0.6)
        reference_flops = [5, 50, 200]
        size_handles = [
            ax.scatter([], [], s=float(bubble_size(value)),
                       facecolor='#EDF2F6', edgecolor='#8797A5', linewidth=0.8)
            for value in reference_flops
        ]
        fig.legend(handles=size_handles, labels=[str(value) for value in reference_flops],
                   title='FLOPs (G) — bubble area', title_fontsize=11,
                   loc='center', bbox_to_anchor=(0.67, 0.105), ncol=3,
                   frameon=False, fontsize=10.5, handletextpad=1.2,
                   columnspacing=2.8, labelspacing=1.2)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true', help='Open an interactive preview.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Export directory (default: the script directory).')
    args = parser.parse_args()
    fig, _ = build_figure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 保存矢量 PDF/SVG 和 300 dpi PNG；从任意目录运行均有稳定输出位置。
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        for extension in ('pdf', 'png', 'svg'):
            output = args.output_dir / f'{OUTPUT_STEM}.{extension}'
            fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.22)
            print(f'Plot saved to {output}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

