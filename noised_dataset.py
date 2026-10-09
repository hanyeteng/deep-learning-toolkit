import cv2
import numpy as np
import os
from pathlib import Path


def add_gaussian_noise(image, variance):
    """
    添加加性高斯白噪声 (AWGN)
    公式: I_noisy = I + N(0, sigma^2)
    """
    # 将方差转换为 0-255 像素值范围内的标准差
    sigma = np.sqrt(variance) * 255.0

    # 生成与图像形状相同的高斯噪声
    noise = np.random.normal(0, sigma, image.shape)

    # 将图像转换为 float32 进行加法运算，避免 uint8 溢出
    noisy_img = image.astype(np.float32) + noise

    # 截断到 0-255 范围并转回 uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def add_speckle_noise(image, variance):
    """
    添加乘性散斑噪声 (Speckle Noise)
    公式: I_noisy = I + I * N(0, sigma^2)
    """
    sigma = np.sqrt(variance)

    # 生成均值为0，标准差为 sigma 的高斯分布矩阵
    noise = np.random.normal(0, sigma, image.shape)

    img_float = image.astype(np.float32)
    # 乘性叠加
    noisy_img = img_float + img_float * noise

    # 截断到 0-255 范围并转回 uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def process_dataset(input_folder, output_base_folder, variance=0.01):
    """
    遍历文件夹，为所有图片添加两种噪声并分别保存
    """
    input_dir = Path(input_folder)
    out_dir_gaussian = Path(output_base_folder) / f"Gaussian_var_{variance}"
    out_dir_speckle = Path(output_base_folder) / f"Speckle_var_{variance}"

    # 创建输出子文件夹
    out_dir_gaussian.mkdir(parents=True, exist_ok=True)
    out_dir_speckle.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    # 获取所有图片路径
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions]
    total_imgs = len(image_paths)

    print(f"找到 {total_imgs} 张图片。开始处理 (方差: {variance})...")

    for i, img_path in enumerate(image_paths, 1):
        # 读取图像 (保持原通道数：灰度图读为单通道，彩图读为三通道)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"警告: 无法读取图像 {img_path.name}，跳过。")
            continue

        # 生成噪声图像
        img_gaussian = add_gaussian_noise(img, variance)
        img_speckle = add_speckle_noise(img, variance)

        # 构建保存路径，保持原文件名
        save_path_gaussian = out_dir_gaussian / img_path.name
        save_path_speckle = out_dir_speckle / img_path.name

        # 保存图像
        cv2.imwrite(str(save_path_gaussian), img_gaussian)
        cv2.imwrite(str(save_path_speckle), img_speckle)

        # 简单进度打印
        if i % 100 == 0 or i == total_imgs:
            print(f"进度: {i}/{total_imgs} 完成.")

    print("所有图像处理完毕！")


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 原始数据集文件夹路径
    INPUT_FOLDER = r"D:\document\DATransNet\data\IRSTD-1k\images"

    # 处理后保存的总文件夹路径
    OUTPUT_FOLDER = r"D:\document\DATransNet\data\IRSTD-1k_noise\images"

    # 设置噪声方差 (您可以依次运行 0.01 和 0.05 来生成两组对比数据)
    NOISE_VARIANCE = 0.01
    # ===========================================

    process_dataset(INPUT_FOLDER, OUTPUT_FOLDER, variance=NOISE_VARIANCE)

