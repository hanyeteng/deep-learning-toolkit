import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from torch.utils.data import DataLoader
from net import Net  # 你项目中的网络定义
from utils.datasets2 import InferenceSetLoader
from utils.utils import Normalized, PadImg  # 你已有的辅助函数
import cv2
import math
import matplotlib.ticker as ticker
from utils.datasets2 import (
    NUDTSIRSTSetLoader,
    IRSTD1KSetLoader,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -------------------------------
# 参数设置
# -------------------------------
parser = argparse.ArgumentParser(description="ROC Curve Comparison for IRSTD models")
parser.add_argument("--model_names", nargs='+',
                    # default=['FSGPNet', 'DATransNet', 'UIUNet', 'DNANet', 'ISTDU-Net', 'ALCNet', 'RDIAN', 'ACM', 'SCTransNet'])
                    # default=['ADGFNet', 'ADGFNet-Lite', 'AGPCNet', 'UIUNet', 'DNANet', 'ISTDU-Net', 'ALCNet', 'ACM'])
                    # default=['ADGFNet', 'ADGFNet-Lite', 'MDAFNet', 'GSFANet', 'DATransNet', 'UIUNet', 'ALCNet', 'DNANet', 'ISTDU-Net'])
                    default=['ADGFNet', 'ADGFNet-Lite'])
                    # default=['ADGFNet', 'ADGFNet-Lite', 'MDAFNet', 'GSFANet', 'GSFANet','GSFANet','GSFANet','GSFANet','GSFANet',])
# default=['MTU_Net', 'AGPCNet', 'SCTransNet'])
# default=['FSGPNet', 'DATransNet', 'SCTransNet'])
parser.add_argument("--pth_dirs", nargs='+', default=[
    # 'NUDT-SIRST/FSGPNet.pth.tar',
    'NUDT-SIRST/ADGFNet_1160.pth.tar',
    'NUDT-SIRST/ADGFNet-Lite_880.pth.tar',
    # 'NUDT-SIRST/MDAFNet_best.pth.tar',
    # 'NUDT-SIRST/GSFANet_400.pth.tar',
    # # 'NUDT-SIRST/RDIAN_400.pth.tar',
    # 'NUDT-SIRST/DATransNet_229.pth.tar',
    # # 'NUDT-SIRST/SCTransNet_500.pth.tar',
    # 'NUDT-SIRST/UIUNet_400.pth.tar',
    # 'NUDT-SIRST/ALCNet_680.pth.tar',
    # 'NUDT-SIRST/DNANet_232.pth.tar',
    # 'NUDT-SIRST/ISTDU-Net_400.pth.tar',
    # 'NUDT-SIRST/RDIAN_400.pth.tar',
    # 'NUDT-SIRST/ACM_550.pth.tar',
    # 'NUDT-SIRST/MTU_Net_301.pth.tar',
    # # 'IRSTD-1K/FSGPNet.pth.tar',
    # 'IRSTD-1K/ADGFNet_354.pth.tar',
    # 'IRSTD-1K/ADGFNet-Lite_246.pth.tar',
    # 'IRSTD-1K/MDAFNet_best.pth.tar',
    # 'IRSTD-1K/GSFANet_371.pth.tar',
    # # 'IRSTD-1K/RDIAN_400.pth.tar',
    # 'IRSTD-1K/DATransNet_258.pth.tar',
    # # 'IRSTD-1K/SCTransNet_300.pth.tar',
    # 'IRSTD-1K/UIUNet_400.pth.tar',
    # 'IRSTD-1K/ALCNet_235.pth.tar',
    # 'IRSTD-1K/DNANet_400.pth.tar',
    # 'IRSTD-1K/ISTDU-Net_400.pth.tar',
    # # 'IRSTD-1K/RDIAN_400.pth.tar',
    # # 'IRSTD-1K/ACM_399.pth.tar',
    # # 'IRSTD-1K/MTU_Net_301.pth.tar',
])
parser.add_argument("--dataset_dir", default='./data', type=str)
parser.add_argument("--dataset_name", default='NUDT-SIRST', type=str)
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--save_log", default='./log5/', type=str)
parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument("--save_plot", default='./roc_results4/roc3.pdf', type=str)

opt = parser.parse_args()


# -------------------------------
# 绘制多模型ROC对比
# -------------------------------
def evaluate_and_plot():
    plt.figure(figsize=(7, 6))
    # colors = ['r', 'g', 'b', 'm', 'c', 'y']
    colors = ['red', 'blue', 'green', 'orange', '#76B7B2', 'purple', 'brown', 'pink', 'gray', 'olive', ]
    # 添加线型列表
    linestyles = ['-', '--', '-.', ':']

    for i, (model_name, pth_dir) in enumerate(zip(opt.model_names, opt.pth_dirs)):
        print(f"\n🔹 Evaluating {model_name} ...")
        # --- 数据加载 ---
        if model_name == 'DATransNet':
            if opt.dataset_name == "NUDT-SIRST":
                test_set = NUDTSIRSTSetLoader(base_dir=os.path.join(opt.dataset_dir, 'NUDT-SIRST'), mode='test')
            elif opt.dataset_name == "IRSTD-1K":
                test_set = IRSTD1KSetLoader(base_dir=os.path.join(opt.dataset_dir, 'IRSTD-1K'), mode='test')
        else:
            test_set = InferenceSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, opt.img_norm_cfg)
        test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=1, shuffle=False)
        sample_img, _, _ = next(iter(test_loader))
        patch_size = sample_img.shape[-1]

        # --- 模型加载 ---
        net = Net(model_name=model_name, mode='test', patch_size=patch_size).to(opt.device)
        checkpoint_path = os.path.join(opt.save_log, pth_dir)
        try:
            if model_name == 'GSFANet':
                # state_dict = torch.load(checkpoint_path, map_location=opt.device)['state_dict']
                ckpt = torch.load(checkpoint_path, map_location=opt.device)
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                # state_dict = ckpt['state_dict']
                state_dict = {'model.' + k: v for k, v in state_dict.items()}
                # print("\nPretrained model state_dict:")
                # for key in list(state_dict.keys()):
                #     print(key)
                net.load_state_dict(state_dict)
            else:
                ckpt = torch.load(checkpoint_path, map_location=opt.device)
                net.load_state_dict(ckpt['state_dict'])
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            continue
        net.eval()

        preds_all, gts_all = [], []

        # --- 推理并收集预测 ---
        for img, size, name in tqdm(test_loader, desc=f"Inference {model_name}"):
            img = img.to(opt.device)
            with torch.no_grad():
                pred = net(img)
                pred = pred[:, :, :size[0], :size[1]]
                # pred = torch.sigmoid(pred)  # 确保概率输出
                pred_np = pred.squeeze().cpu().numpy()

            # 加载GT
            gt_path = os.path.join(opt.dataset_dir, opt.dataset_name, 'masks', f"{name[0]}.png")
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            gt = gt.astype(np.float32) / 255.0
            pred_np = cv2.resize(pred_np, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            preds_all.append(pred_np.flatten())
            gts_all.append(gt.flatten())

        preds_all = np.concatenate(preds_all)
        gts_all = np.concatenate(gts_all)

        gts_all = (gts_all > 0.5).astype(np.uint8)  # 确保GT为0或1

        # --- 计算 ROC ---
        fpr, tpr, _ = roc_curve(gts_all, preds_all)
        roc_auc = auc(fpr, tpr)
        print(f"✅ {model_name} AUC = {roc_auc:.4f}")
        roc_auc = math.floor(roc_auc * 1000) / 1000

        if model_name == 'ADGFNet':
            zorder = 100
        elif model_name == 'ADGFNet-Lite':
            zorder = 99
        else:
            zorder = 1

        plt.plot(fpr, tpr,
            color=colors[i % len(colors)],
            linestyle=linestyles[(i // len(colors)) % len(linestyles)],
            label=f"{model_name} (AUC={roc_auc:.3f})",
            lw=2,
            zorder=zorder,
        )

        # plt.plot(fpr, tpr,
        #          color=colors[i % len(colors)],
        #          linestyle=linestyles[i // len(colors)],  # 循环使用线型
        #          # label=f"{model_name}", lw=2)
        #          label=f"{model_name} (AUC={roc_auc:.3f})", lw=2)

    # plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve Comparison on {opt.dataset_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    # plt.tight_layout()

    x_right = 0.0001
    plt.xlim(0, x_right)
    plt.ylim(0, 1.0)

    # X 轴刻度：从 0 开始，到 0.0002 结束，步长设为 0.00005
    x_step = x_right/4
    x_ticks = np.arange(0, x_right + x_step, x_step)
    plt.xticks(x_ticks)

    # Y 轴刻度：从 0 开始，到 1.0 结束，步长设为 0.2
    y_step = 0.2
    y_ticks = np.arange(0, 1.0 + y_step, y_step)
    plt.yticks(y_ticks)
    os.makedirs(os.path.dirname(opt.save_plot), exist_ok=True)
    # plt.savefig(opt.save_plot, dpi=1000, bbox_inches='tight')
    plt.savefig(opt.save_plot, format='pdf', bbox_inches='tight')
    print(f"\nROC 图已保存至: {opt.save_plot}")
    plt.show()


# -------------------------------
# 主函数入口
# -------------------------------
if __name__ == '__main__':
    evaluate_and_plot()
