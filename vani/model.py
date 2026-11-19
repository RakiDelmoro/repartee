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
        mask = torch.triu(
            torch.ones(CODEBOOKS + 1, CODEBOOKS + 1, dtype=torch.bool), diagonal=1
        )
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
        # Current-frame text -> depth transformer.  Zero-init so an old P1
        # checkpoint starts exactly where it left off and only learns the new path.
        self.text_to_depth = nn.Linear(d, depth_dim)
        self.text_pos = nn.Parameter(torch.zeros(depth_dim))
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
        nn.init.zeros_(self.text_to_depth.weight)
        nn.init.zeros_(self.text_to_depth.bias)
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

    def text_depth(self, text: torch.Tensor) -> torch.Tensor:
        """Text ids [B,T] or [B] -> depth vectors [B,T,1,D_depth]."""
        if text.dim() == 1:
            text = text[:, None]
        return self.text_to_depth(self.text_embed(text)).unsqueeze(2)

    def audio_logits(self, hidden: torch.Tensor, target_audio: torch.Tensor,
                     text_cond: torch.Tensor) -> torch.Tensor:
        """Depth prediction for one frame; text is the current frame's word."""
        start = torch.full_like(target_audio[:, :, :1], CODEBOOK_SIZE)
        prev = torch.cat([start, target_audio[:, :, :-1]], dim=2)  # [B,T,8]
        base = self.to_depth(hidden).unsqueeze(2)
        text = base + self.text_depth(text_cond) + self.text_pos  # [B,T,1,D]
        sound = base + self.code_embed(prev) + self.code_pos.weight[None, None, :, :]
        depth_in = torch.cat([text, sound], dim=2)  # [B,T,1+8,D]
        b, t, k, dim = depth_in.shape
        depth_out = self.depth(depth_in.reshape(b * t, k, dim))
        depth_out = depth_out.reshape(b, t, k, dim)
        return torch.stack(
            [self.audio_head[cb](depth_out[:, :, 1 + cb]) for cb in range(CODEBOOKS)],
            dim=2,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                text_cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced logits.

        inputs:    [B, T, 17] previous-frame codes (already shifted)
        targets:   [B, T, 17] current-frame codes
        text_cond: [B, T] current-frame text ids; default is targets[:,:,0].
                   This is the channel that lets a word control its own audio
                   frame and is what P3 later forces.
        returns text_logits [B,T,V], audio_logits [B,T,8,2048]
        """
        hidden = self.temporal_hidden(inputs)
        text_logits = self.lm_head(hidden)
        if text_cond is None:
            text_cond = targets[:, :, 0]

        target_audio = targets[:, :, 1 : 1 + CODEBOOKS]
        audio_logits = self.audio_logits(hidden, target_audio, text_cond)
        return text_logits, audio_logits


def load_vani_state(model: VaniM, path) -> dict:
    """Load a checkpoint, tolerating the new text_to_depth path.

    Older P1 checkpoints predate the current-frame text channel.  Missing keys
    are allowed only for that zero-initialised projection.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    bad = [k for k in missing
           if not (k.startswith("text_to_depth.") or k == "text_pos")]
    if bad:
        raise RuntimeError(f"missing checkpoint keys: {bad}")
    if missing:
        print(f"initialised new text channel: {missing}", flush=True)
    return state
