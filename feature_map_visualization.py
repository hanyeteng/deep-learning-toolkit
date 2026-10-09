import argparse
import torch
from torch.utils.data import DataLoader
import os
from net import Net
from evaluation.mIoU import mIoU
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP
from evaluation.mIoU import SamplewiseSigmoidMetric
import pandas as pds  # 修改缩写避免冲突
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils.datasets_noise import (
    NUDTSIRSTSetLoader,
    IRSTD1KSetLoader,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test and visualize feature maps")
    parser.add_argument("--model_name", type=str, default='UIUNet')
    parser.add_argument("--dataset_names", nargs='+', default=['IRSTD-1K'])
    parser.add_argument("--patchSize", type=int, default=512)
    parser.add_argument("--dataset_dir", type=str, default='./data')
    parser.add_argument("--weight_path", type=str, default='./log5/IRSTD-1K/UIUNet_400.pth.tar')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_img", default=True, type=bool)
    parser.add_argument("--save_img_dir", type=str, default='./results_noise3/')
    parser.add_argument("--save_feature_maps", default=True, type=bool)
    parser.add_argument("--noise_type", type=str, default='gaussian')
    parser.add_argument("--noise_variance", type=float, default=0.001)
    return parser.parse_args()


def load_dataset_noise(dataset_name, dataset_dir, mode='test', noise_type='none', noise_variance=0.0):
    if dataset_name == "NUDT-SIRST":
        return NUDTSIRSTSetLoader(base_dir=os.path.join(dataset_dir, 'NUDT-SIRST'), mode=mode,
                                  noise_type=noise_type, noise_variance=noise_variance)
    elif dataset_name == "IRSTD-1K":
        return IRSTD1KSetLoader(base_dir=os.path.join(dataset_dir, 'IRSTD-1K'), mode=mode,
                                noise_type=noise_type, noise_variance=noise_variance)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def save_pure_feature_map(feat, save_path, name):
    """
    仅保存图像矩阵，不带坐标轴、标题和图例
    """
    # 将特征图从 Tensor 转为 Numpy
    feat = feat.detach().cpu()

    # 对通道取平均 (C, H, W) -> (H, W)
    if feat.shape[1] > 1:
        feat_map = feat.mean(dim=1).squeeze(0).numpy()
    else:
        feat_map = feat.squeeze(0).squeeze(0).numpy()

    # 归一化到 0-1 方便着色
    f_min, f_max = feat_map.min(), feat_map.max()
    if f_max > f_min:
        feat_map = (feat_map - f_min) / (f_max - f_min)

    # 构造文件名并保存
    file_name = f"{name}.png"
    full_path = os.path.join(save_path, file_name)

    # 使用 imsave 直接保存原始矩阵，使用 viridis 映射（也可以换成 jet 或 gray）
    plt.imsave(full_path, feat_map, cmap='viridis')


def test_model(args):
    model = Net(model_name=args.model_name, mode='test', patch_size=args.patchSize).cuda()
    state_dict = torch.load(args.weight_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # --- Hook 设置 ---
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output

        return hook

    # 注册解码器 Hook
    for name, module in model.named_modules():
        if name.endswith('decoder4'): module.register_forward_hook(get_features('dec4'))
        if name.endswith('decoder3'): module.register_forward_hook(get_features('dec3'))
        if name.endswith('decoder2'): module.register_forward_hook(get_features('dec2'))
        if name.endswith('decoder1'): module.register_forward_hook(get_features('dec1'))

    results = []

    for dataset_name in args.dataset_names:
        print(f"\nProcessing: {dataset_name}")
        test_set = load_dataset_noise(dataset_name, args.dataset_dir, mode='test',
                                      noise_type=args.noise_type, noise_variance=args.noise_variance)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)  # Windows建议0

        eval_mIoU = mIoU()
        nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
        eval_PD_FA = PD_FA()
        eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

        for idx, (img, gt_mask, size, img_path) in enumerate(test_loader):
            img_name_raw = img_path[0]
            # 创建该图片的独立存储文件夹
            img_feat_dir = os.path.join(args.save_img_dir, dataset_name, args.model_name, 'features', img_name_raw)
            if args.save_feature_maps and not os.path.exists(img_feat_dir):
                os.makedirs(img_feat_dir)

            with torch.no_grad():
                img = img.cuda()
                pred = model(img)

                # 保存各个层级的特征图
                if args.save_feature_maps:
                    for feat_name, feat_tensor in features.items():
                        save_pure_feature_map(feat_tensor, img_feat_dir, feat_name)
                    # 保存最终输出图
                    save_pure_feature_map(pred, img_feat_dir, 'final_pred')

                pred = pred[:, :, :size[0], :size[1]]

            gt_mask = gt_mask[:, :, :size[0], :size[1]]

            # 指标更新 (重命名 pd 为 pd_val 避免覆盖 pandas)
            eval_mIoU.update((pred > args.threshold).cpu(), gt_mask)
            nIoU_metric.update(pred, gt_mask)
            eval_PD_FA.update(pred[0, 0, :, :].cpu().numpy(), gt_mask[0, 0, :, :].numpy(), size)
            eval_mIoU_P_R_F.update(gt_mask[0, 0, :, :].numpy(), pred[0, 0, :, :].cpu().numpy())

            pixAcc, mIoU_val = eval_mIoU.get()
            nIoU_val = nIoU_metric.get()
            pd_val, fa_val = eval_PD_FA.get()
            _, recall, precision, fscore = eval_mIoU_P_R_F.get()

            results.append({
                "Image Name": img_name_raw,
                "mIoU": mIoU_val,
                "nIoU": nIoU_val,
                "Pd": pd_val,
                "Fa": fa_val,
                "F-score": fscore
            })

            # 重置单图指标
            eval_mIoU.reset();
            eval_PD_FA.reset();
            eval_mIoU_P_R_F.reset();
            nIoU_metric.reset()

            # 保存二值化结果
            if args.save_img:
                mask_dir = os.path.join(args.save_img_dir, dataset_name, args.model_name, 'masks')
                if not os.path.exists(mask_dir): os.makedirs(mask_dir)
                img_save = transforms.ToPILImage()(((pred[0, 0, :, :] > args.threshold).float()).cpu())
                img_save.save(os.path.join(mask_dir, img_name_raw + '.png'))

    # 使用 pds.DataFrame 确保不会报错
    df = pds.DataFrame(results)
    output_excel_path = args.model_name + '_results.xlsx'
    df.to_excel(output_excel_path, index=False)
    print(f"Finished. Data saved to {output_excel_path}")


if __name__ == "__main__":
    args = parse_args()
    test_model(args)