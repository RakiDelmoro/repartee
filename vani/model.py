"""Vani-M P1: Qwen3 temporal backbone + 12-layer depth transformer.

One temporal step per 80 ms.  P1 trains the agent mouth only: text + 8 agent
Mimi codebooks are inputs/targets; the 8 user codebooks are zeroed and ignored.
"""

from __future__ import annotations

import torch
from torch import nn

from .data.mimi import CODEBOOKS

TEXT_PAD = 151_643
CODEBOOK_SIZE = 2048
AGENT_STREAMS = 1 + CODEBOOKS
ALL_STREAMS = 1 + 2 * CODEBOOKS


class DepthTransformer(nn.Module):
    def __init__(self, dim: int = 768, layers: int = 12, heads: int = 8, ff: int = 3072):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        mask = torch.triu(torch.ones(CODEBOOKS, CODEBOOKS, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, codebooks, dim] -> [B*T, codebooks, dim]."""
        return self.norm(self.net(x, mask=self.causal_mask))


class VaniM(nn.Module):
    def __init__(self, qwen_path: str, depth_dim: int = 768, depth_layers: int = 12,
                 depth_heads: int = 8, use_user: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import Qwen3ForCausalLM

        self.qwen = Qwen3ForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.temporal = self.qwen.model
        self.lm_head = self.qwen.lm_head
        self.text_embed = self.temporal.embed_tokens
        self.use_user = use_user
        d = self.temporal.config.hidden_size

        self.stream_embed = nn.Embedding(ALL_STREAMS, d)
        self.audio_embed = nn.ModuleList(
            [nn.Embedding(CODEBOOK_SIZE, d) for _ in range(2 * CODEBOOKS)]
        )
        self.to_depth = nn.Linear(d, depth_dim)
        self.code_embed = nn.Embedding(CODEBOOK_SIZE + 1, depth_dim)  # + start token
        self.code_pos = nn.Embedding(CODEBOOKS, depth_dim)
        self.depth = DepthTransformer(depth_dim, depth_layers, depth_heads)
        self.audio_head = nn.ModuleList(
            [nn.Linear(depth_dim, CODEBOOK_SIZE) for _ in range(CODEBOOKS)]
        )
        nn.init.normal_(self.stream_embed.weight, std=0.02)
        for emb in self.audio_embed:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.code_embed.weight, std=0.02)
        nn.init.normal_(self.code_pos.weight, std=0.02)
        self.depth.causal_mask = self.depth.causal_mask  # keep buffer on module
        self.to(dtype)

    def temporal_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [B, T, 17] shifted codes -> hidden [B, T, D]."""
        text = inputs[:, :, 0]
        emb = self.text_embed(text) + self.stream_embed.weight[0]
        for cb in range(CODEBOOKS):
            emb = emb + self.audio_embed[cb](inputs[:, :, 1 + cb])
            emb = emb + self.stream_embed.weight[1 + cb]
        if self.use_user:
            for cb in range(CODEBOOKS):
                emb = emb + self.audio_embed[CODEBOOKS + cb](inputs[:, :, 1 + CODEBOOKS + cb])
                emb = emb + self.stream_embed.weight[1 + CODEBOOKS + cb]
        out = self.temporal(inputs_embeds=emb, use_cache=False, return_dict=True)
        return out.last_hidden_state

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced logits.

        inputs:  [B, T, 17] previous-frame codes (already shifted)
        targets: [B, T, 17] current-frame codes
        returns text_logits [B,T,V], audio_logits [B,T,8,2048]
        """
        hidden = self.temporal_hidden(inputs)
        text_logits = self.lm_head(hidden)

        target_audio = targets[:, :, 1 : 1 + CODEBOOKS]
        start = torch.full_like(target_audio[:, :, :1], CODEBOOK_SIZE)
        prev = torch.cat([start, target_audio[:, :, :-1]], dim=2)  # [B,T,8]

        depth_in = self.to_depth(hidden).unsqueeze(2)
        depth_in = depth_in + self.code_embed(prev) + self.code_pos.weight[None, None, :, :]
        b, t, k, dim = depth_in.shape
        depth_out = self.depth(depth_in.reshape(b * t, k, dim))
        depth_out = depth_out.reshape(b, t, k, dim)
        audio_logits = torch.stack(
            [self.audio_head[cb](depth_out[:, :, cb]) for cb in range(CODEBOOKS)], dim=2
        )
        return text_logits, audio_logits
