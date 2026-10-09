import argparse
import torch
import time
import os
from thop import profile
from net import Net
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="IRSTD Model Benchmark")

parser.add_argument("--model_names",
                    default=['ALCNet'],
                    # default=['FSGPNet_N123', 'FSGPNet_N23', 'FSGPNet_N3', 'FSGPNet'],
                    # default=['ACM','ALCNet','AGPCNet', 'DNANet','UIUNet','RDIAN','ISTDU_Net',
                    #          'SCTransNet','DATransNet','GSFANet','FMNet','FSGPNet'],
                    nargs='+')
# parser.add_argument("--model_names",
#                     default=['ACM','ALCNet','DNANet','ISNet','RISTDnet','UIUNet','U-Net','RDIAN','ISTDU-Net'],
#                     nargs='+')

parser.add_argument("--img_size", default=512, type=int)
parser.add_argument("--repeat", default=200, type=int)

opt = parser.parse_args()


def benchmark(model, input_tensor, repeat=200):

    model.eval()

    # ---------- warm up ----------
    with torch.no_grad():
        for _ in range(30):
            _ = model(input_tensor)

    torch.cuda.synchronize()

    # ---------- timing ----------
    start = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)

    torch.cuda.synchronize()

    end = time.time()

    latency = (end - start) / repeat
    fps = 1.0 / latency

    return latency, fps


if __name__ == "__main__":

    results = []

    input_tensor = torch.randn(1,1,opt.img_size,opt.img_size).cuda()

    for model_name in opt.model_names:
        print("\nTesting:", model_name)

        # 1. 先创建一个模型用于计算 FLOPs 和 Params
        model = Net(model_name, mode='test', patch_size=opt.img_size).cuda()
        model.eval()

        try:
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        except Exception as e:
            print(f"Warning: FLOPs calculation failed for {model_name}: {str(e)}")
            flops = 0
            params = sum(p.numel() for p in model.parameters())

        # --- 关键修改：重新实例化模型，清除 thop 的 hooks ---
        del model  # 释放旧模型
        torch.cuda.empty_cache()
        model = Net(model_name, mode='test', patch_size=opt.img_size).cuda()
        model.eval()
        # ----------------------------------------------

        # ---------- GPU Memory ----------
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)

        mem = torch.cuda.max_memory_allocated() / 1024 ** 2

        # ---------- Inference Time ----------
        latency, fps = benchmark(model, input_tensor, opt.repeat)
        # ... 后续打印代码保持不变 ...
        params_m = params / 1e6
        flops_g = flops / 1e9
        latency_ms = latency * 1000

        print("Params: %.2f M" % params_m)
        print("FLOPs: %.2f G" % flops_g)
        print("Latency: %.3f ms" % latency_ms)
        print("FPS: %.2f" % fps)
        print("Memory: %.2f MB" % mem)

        results.append([model_name, params_m, flops_g, latency_ms, fps, mem])

    print("\n================ Benchmark Results ================\n")

    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model","Params(M)","FLOPs(G)","Latency(ms)","FPS","Mem(MB)"
    ))

    for r in results:
        print("{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(*r))
