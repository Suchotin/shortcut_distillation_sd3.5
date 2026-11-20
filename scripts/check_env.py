"""GPU / CUDA / PyTorch sanity check for the training box (run with the GPU attached).

Checks: driver + CUDA runtime of the torch build, compute capability (Hopper = sm_90), bf16,
that the build ships kernels for this GPU, bf16 matmul throughput, and which fused attention
kernels (flash / cuDNN / mem-efficient) SDPA can use at SD3.5 shapes.
"""

import subprocess
import sys
import time

import torch


def main():
    ok = True
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | CUDA runtime {torch.version.cuda}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print(f"nvidia-smi: {smi}")
    except FileNotFoundError:
        print("nvidia-smi not found")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False. Most likely the host driver is older than the "
              "CUDA runtime of this torch build. Reinstall torch for CUDA 12: "
              "TORCH_CU=cu126 bash setup_env.sh install")
        sys.exit(1)

    p = torch.cuda.get_device_properties(0)
    cc = f"sm_{p.major}{p.minor}"
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB, {cc}, {p.multi_processor_count} SMs")
    archs = torch.cuda.get_arch_list()
    if cc not in archs:
        print(f"WARN: {cc} not in torch arch list {archs} (will rely on PTX JIT)")
    if not torch.cuda.is_bf16_supported():
        print("FAIL: bf16 not supported")
        ok = False

    # bf16 matmul throughput (Hopper-class GPUs: expect several hundred TFLOPS)
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0, it = time.time(), 50
    for _ in range(it):
        a @ b
    torch.cuda.synchronize()
    tflops = 2 * n**3 * it / (time.time() - t0) / 1e12
    print(f"bf16 matmul {n}x{n}: {tflops:.0f} TFLOPS")

    # SDPA kernels at SD3.5 Medium shapes (24 heads x 64, 512^2 image + 333 text tokens)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(8, 24, 1357, 64, device="cuda", dtype=torch.bfloat16)
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION),
                     ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION)]:
        try:
            with sdpa_kernel([be]):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            print(f"SDPA {name}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"SDPA {name}: unavailable ({type(e).__name__})")

    import accelerate
    import diffusers
    import peft
    import transformers

    print(f"diffusers {diffusers.__version__} | transformers {transformers.__version__} | "
          f"peft {peft.__version__} | accelerate {accelerate.__version__}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
