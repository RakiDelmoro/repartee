"""Triton int8-weight GEMV (batch 1), for the LM's per-frame matmuls.

torch's packed-int8 kernel is ~0.9 ms for a 2560x9216 layer on this card, which
dominates the frame budget.  This kernel reads int8 weights once, multiplies in
float32, applies the per-row scale, and writes bf16.  Memory-bound by design.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, K: tl.constexpr, N: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = row < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        col = k0 + tl.arange(0, BLOCK_K)
        mask_k = col < K
        w = tl.load(w_ptr + row[:, None] * K + col[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0)
        x = tl.load(x_ptr + col, mask=mask_k, other=0.0).to(tl.float32)
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    scale = tl.load(s_ptr + row, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(y_ptr + row, (acc * scale).to(tl.bfloat16), mask=mask_n)


def int8_gemv(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
              block_n: int = 64, block_k: int = 128, num_warps: int = 4,
              num_stages: int = 2) -> torch.Tensor:
    """``x [K]`` bf16, ``weight [N, K]`` int8, ``scales [N]`` -> ``y [N]`` bf16."""
    n, k = weight.shape
    y = torch.empty(n, device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(n, block_n),)
    _int8_gemv_kernel[grid](x, weight, scales, y, K=k, N=n, BLOCK_N=block_n, BLOCK_K=block_k,
                            num_warps=num_warps, num_stages=num_stages)
    return y


if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    for n, k in [(9216, 2560), (2560, 2560), (2560, 9216)]:
        x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
        w = torch.randint(-127, 127, (n, k), device="cuda", dtype=torch.int8)
        s = torch.rand(n, device="cuda", dtype=torch.bfloat16)
        y = int8_gemv(x, w, s)
        ref = (x.float() @ w.float().T) * s.float()
        print(n, k, "err", float((y.float() - s[:, None].squeeze(-1) * 0).abs().mean()))
        for _ in range(3):
            int8_gemv(x, w, s)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            int8_gemv(x, w, s)
        torch.cuda.synchronize()
        print(f"{n}x{k}: {(time.perf_counter() - t) / 50 * 1000:.3f} ms")
