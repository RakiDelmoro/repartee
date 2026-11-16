# Running Haru-4B on 8 GB: 4-bit LM, fp16 audio

Goal: fit the 10.5 GB checkpoint into the 8 GB Quadro RTX 4000 without changing what the
model says, as far as our tests can tell. "Lossless 4-bit" is not a thing; the target is
**quality-neutral on our gates**, and the audio path is not quantized.

## Decision

| part | precision | why |
|---|---|---|
| `lm.*` Linear layers | **4-bit NF4, double quant** | ~6 GB saved; the LM has the redundancy |
| `lm.embed_tokens` (tied output) | bf16 | text logits are computed from this weight; kept off the dequant path |
| `head.*` (MeanFlow) | bf16 | its output *is* the audio latent; tiny, no redundancy |
| `codec.*` (convs) | bf16 | output is the waveform; conv quantization is fragile and saves little |
| audio projections | bf16 | 64 → 2560, between audio and LM |
| RMSNorm, rotary buffers, `latent_scale/bias` | float32 | explicit upcasts in the model; they set numeric scale |

Fallback ladder: if 4-bit fails a quality gate, raise the LM to 8-bit (same loader,
`--lm-bits 8`); if it still fails, run bf16 on rented hardware. Do not quantize the audio
path to compensate.

## Loader

`research/haru_quant.py` implements the order that keeps the 8 GB card from holding a
bf16 copy:

1. load the checkpoint on **CPU** in bf16 (`load_model(..., device="cpu")`);
2. replace every `nn.Linear` under `model.lm` with `bitsandbytes` `Linear4bit`
   (`compute_dtype=float16`, `quant_type="nf4"`, `compress_statistics=True`), copying the
   bf16 weights in;
3. cast the remaining **parameters** to bf16 (the checkpoint's dtype); leave buffers
   float32;
4. `model.to("cuda")`: bnb quantizes the 4-bit weights during the move, the rest arrives
   as bf16.

fp16 was tried first and is wrong for this checkpoint: the real model produced non-finite
logits at `start()` (`multinomial` assertion on inf/NaN) with fp16 parameters and bf16
weights converted, while the same loader in bf16 runs. bf16 is also what the checkpoint
was trained in; Turing runs it, slower than native bf16 hardware but correctly.

`Linear4bit` only replaces `nn.Linear`; conv stacks, embeddings and norms are untouched.
The model's per-frame text logits still multiply the fp16 embedding weight, so no
dequantization is added to the 7.5 Hz hot path.

## What must pass

Run the same clips through bf16 (reference on rented hardware or CPU) and the 4-bit
build:

| gate | target |
|---|---|
| VRAM after load | < 7.5 GB, no OOM |
| ms/frame (median) | report; live only if ≤ 133 ms with margin |
| ASR word recall of model speech | ≥ 0.95 vs bf16 |
| injected-text recall | ≥ 0.95 (the feature's contract) |
| duplicate/truncated word at the seam | 0 |
| F0 drift vs bf16 | bounded; no register jump |
| duplex behaviour | still overlaps, stops, backchannels |

`research/haru_inject.py --lm-bits 4` runs the five scenarios and ASR-gates the injected
sentences; `research/haru_quant.py --frames 40 --speak-first --float-dtype bf16` is the
load/VRAM/latency probe.

### First measurement

5-frame probe on the RTX 4000 (8 GB), 4-bit LM + bf16 audio path:

| measure | value |
|---|---|
| VRAM after load | **5.41 GB** |
| quantized Linear layers | 248 |
| load + quantize | 278 s |
| frame time (median / p90) | **236 / 378 ms** |
| text | `Hello` |

Fits with room to spare. 236 ms/frame is **1.8× the 133 ms budget**, so this card is an
offline test bed, not a live server, unless the hybrid-attention fallback kernels are
replaced.

### Quality: 4-bit fails the audio gate

The model's text stream was exact under 4-bit, but the audio was not. Forced text that
bf16 renders as "Light from the sun takes about eight minutes to reach Earth." came out
under 4-bit as "I don't know. Do I?" (Qwen3-ASR 1.7B). The LM backbone is too sensitive
at NF4 for speech, even though the head and codec stayed bf16.

8-bit is not a local CUDA fallback: with `bitsandbytes` it loads at **7.27 GB** and runs
at **1.56 s/frame** (`MatMul8bitLt` casts bf16 to fp16 on every call); it is slower and
larger than 4-bit, not better. Use bf16 on CPU (or rented hardware) for audio quality on
CUDA; keep 4-bit only as a memory/latency probe.

### External confirmation: 8-bit MLX is the viable precision

The community conversion [`nano16/Haru-4B-mlx-8bit`](https://huggingface.co/nano16/Haru-4B-mlx-8bit)
measures the same split, independently:

| precision | text top-1 vs bf16 | audio latent cosine | weights | ms/frame (M4 Pro) |
|---|---|---|---|---|
| bf16 | — | — | 10.5 GB | 50.4 |
| **8-bit MLX** | **97.6%** | **0.992** | **6.6 GB** | **34.1** |
| 4-bit | 46.9% | 0.706 | — | — |

Its 4-bit result ("mixed-script nonsense") matches this repo's 4-bit audio failure. The
catch is portability: MLX weights are an Apple-silicon layout and load through
`load_haru_8bit.py`, not the PyTorch `HaruModel`; they cannot run on this CUDA box. On an
M-series Mac (16 GB+), 8-bit MLX is inside the 133 ms budget and is the right artifact;
on CUDA, 8-bit needs a larger GPU than 8 GB.

### 8-bit on this card: quality is solved, 133 ms is not

A custom weight-only int8 path (`research/int8_linear.py` + `research/int8_gemv.py`) keeps
the LM text stream near bf16 and uses a Triton GEMV for the batch-1 matmuls:

| build | ms/frame | VRAM | audio |
|---|---|---|---|
| 4-bit NF4 | 203 | 5.4 GB | broken |
| int8, torch packed kernel | 336 | 7.1 GB | good |
| **int8 + Triton GEMV** | **217** (p90 278) | **7.1 GB** | **good**; ASR matches the text stream |

The profile of the GEMV build shows ~48 ms of GPU compute per frame and a much larger
host-side cost: ~1 500 small ops per frame, with `addmm`/`mm`/`copy_`/`mul` dominating
the CPU side. `torch.compile` is not available on Python 3.14, so the remaining
85 ms must come from fused linear-attention kernels (`fla`, `causal_conv1d`) or a
CUDA graph with a preallocated KV cache. The installs available here do not provide a
working `fla.ops`; that is the next blocker.

So: 8-bit quality is achieved on the 8 GB card; 133 ms is still 1.6x away and needs a
kernel/graph project, not another quantization step.

### 4-bit mitigation, if it is still wanted

The quality failure is in the LM hidden state, so mitigation spends bits where the
error compounds:

1. **Mixed precision** — 4-bit MLP, 8-bit linear-attention projections (the recurrent
   Gated DeltaNet path is the most error-sensitive), embeddings and audio path bf16.
   Memory still fits 8 GB.
2. **Residual low-rank correction** — store `R = W - dequant(W4)` as a small bf16
   low-rank factor per layer; add it after the 4-bit matmul. Cheap in memory, some
   compute.
3. **QAT/distillation** — the only route that can actually recover bf16-like audio at
   4 bits; needs training data and GPU time.
4. **Draft/verify** — run 4-bit for speed, ASR the output, re-render failed phrases on
   the 8-bit path. Trades latency for fidelity.

None of these changes the arithmetic: 4-bit all-LM is 203 ms, 8-bit is 217 ms. The 4-bit
speed advantage is small; 8-bit is the better default on this card.

## Risks

- **Hybrid attention kernels.** Qwen3.5's linear-attention layers currently fall back to
  reference PyTorch (`causal_conv1d`, `flash-linear-attention` are not installed). That,
  not the 4-bit matmul, is the likely latency miss on this card.
- **Turing.** No native bf16, but bf16 compute works through emulation and is required for
  numerical stability; fp16 overflowed the logits. NF4 is supported; GPTQ/AWQ/Marlin
  kernels generally target newer architectures and were not considered.
- **KV cache.** Quantization does not shrink it; long sessions still grow. The
  full-attention layers are every fourth layer, but cap the session length for now.
- **Custom model.** The checkpoint is a `HaruModel`, not a stock HF LM; quantization is a
  loader patch, not a serving flag.
