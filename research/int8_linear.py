"""Weight-only int8 Linear for the LM backbone, using torch's packed-int8 matmul.

No torchao needed: ``torch._weight_int8pack_mm`` is available on this Turing card
(measured 2.1 ms for 2560x9216 at batch 1).  Weights are symmetric per-output-channel
int8 with a float32 scale; activations stay bf16 and the result is cast back.

This is the quality-preserving 8-bit path (unlike NF4): it keeps the LM's text stream
near bf16, which is what the 4-bit build lost.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from int8_gemv import int8_gemv


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.empty(out_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] == 1:
            y = int8_gemv(flat[0], self.weight, self.scales, block_n=32, block_k=256,
                          num_warps=4)[None]
        else:
            y = torch._weight_int8pack_mm(flat, self.weight, self.scales)
        y = y.reshape(*shape[:-1], self.out_features).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        weight = linear.weight.detach().float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        quantized = torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8)
        module = cls(linear.in_features, linear.out_features, linear.bias is not None)
        module.weight.copy_(quantized)
        module.scales.copy_(scales.to(torch.bfloat16))
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module


def replace_lm_linears_int8(model) -> int:
    """Quantize every Linear under ``lm`` (excluding the tied embedding/head)."""
    replaced = 0
    for name, module in list(model.lm.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, Int8Linear):
            continue
        if name.endswith("embed_tokens") or name.endswith("lm_head"):
            continue
        new = Int8Linear.from_linear(module)
        parent = model.lm.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model.lm
        setattr(parent, name.rsplit(".", 1)[-1], new)
        replaced += 1
    model.requires_grad_(False)
    return replaced
