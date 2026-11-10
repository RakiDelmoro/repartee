#!/usr/bin/env python3
"""Evaluate the continuity switch on held-out pairs."""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache

sys.path.insert(0, ".")
sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from mouth_asr import norm, wer

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
OUT = CT / "eval"
SR = 24_000
LIMIT = 20


@torch.no_grad()
def synth(model, proc, codec, lead: str, b_text: str, cut_frames: int):
    conv = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": "Read the following sentence exactly."},
    ]
    prompt = proc.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0]
    ids = torch.cat([
        prompt,
        torch.tensor([model.config.mixed_start_token_id]),
        torch.tensor(proc.tokenizer.encode(lead, add_special_tokens=False)),
        torch.tensor([model.config.text_end_token_id]),
    ]).to(DEVICE)[None]
    out = model.model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
        past_key_values=DynamicCache(config=model.config.text_config),
    )
    hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
    mask = torch.ones_like(ids)

    def feed(e):
        nonlocal hidden, cache, mask
        if e.ndim == 2:
            e = e.unsqueeze(1)
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
        step = model.model.language_model(
            inputs_embeds=e, attention_mask=mask,
            position_ids=torch.tensor([[cache.get_seq_length()]], device=DEVICE),
            past_key_values=cache, use_cache=True,
        )
        hidden, cache = step.last_hidden_state[:, -1], step.past_key_values

    frames = []
    switched = False

    def inject():
        nonlocal switched
        for t in proc.tokenizer.encode(b_text, add_special_tokens=False):
            feed(model.get_input_embeddings()(torch.tensor([[t]], device=DEVICE)))
        feed(model.get_input_embeddings()(torch.tensor([[model.config.text_end_token_id]], device=DEVICE)))
        switched = True

    for i in range(240):
        if i == cut_frames and not switched:
            inject()
        history = torch.stack(frames[-64:]) if frames else None
        codes = model.sample_audio_frame(hidden[0], temperature=0.4, repetition_penalty=1.2, history=history)
        if bool((codes == model.config.audio_eos_code).any()):
            if not switched:
                inject()
                continue
            break
        frames.append(codes)
        feed(model.model.get_audio_out_embeds(codes[None]))
    codes = torch.stack(frames).t()
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    return wav


def b_ratio(hyp: str, b_text: str) -> float:
    h, b = norm(hyp), norm(b_text)
    if not b:
        return 0.0
    prev = [0] * (len(h) + 1)
    for w in b:
        cur = [0]
        for j, x in enumerate(h, 1):
            cur.append(prev[j - 1] + 1 if w == x else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1] / len(b)


def gap_ms(path: Path, switch_s: float) -> int:
    x, sr = sf.read(path, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    hop = int(0.01 * sr); n = len(x) // hop
    rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)); thr = 0.012
    i0 = min(int(switch_s / 0.01), n - 1); lo = i0
    while lo > 0 and rms[lo - 1] < thr:
        lo -= 1
    hi = i0
    while hi < n and rms[hi] < thr:
        hi += 1
    return (hi - lo) * 10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = [m for m in json.load(open(CT / "meta.json")) if m["split"] == "holdout"][:LIMIT]
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()
    sw = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload().eval()
    model = PeftModel.from_pretrained(sw, CT / "lora_v2").merge_and_unload().eval()

    rows = []
    for i, m in enumerate(meta, 1):
        wav = synth(model, proc, codec, m["lead_text"], m["b_text"], m["cut_frames"])
        path = OUT / f"{m['id']}.wav"
        sf.write(path, wav, SR)
        rows.append({"id": m["id"], "lead": m["lead_text"], "b": m["b_text"],
                     "cut_s": m["cut_s"], "s": round(len(wav) / SR, 2), "gap_ms": gap_ms(path, m["cut_s"])})
        print(i, len(meta), m["id"], rows[-1]["gap_ms"], flush=True)

    del model, mouth, base
    torch.cuda.empty_cache()
    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    for r in rows:
        hyp = t.transcribe(OUT / f"{r['id']}.wav").text
        r["hyp"] = hyp
        ratio = b_ratio(hyp, r["b"])
        r["b_recall"] = round(ratio, 3)
        r["b_present"] = ratio == 1.0

    report = {"n": len(rows), "b_present": round(sum(r["b_present"] for r in rows) / len(rows), 3),
              "mean_b_recall": round(sum(r["b_recall"] for r in rows) / len(rows), 4),
              "median_gap_ms": int(np.median([r["gap_ms"] for r in rows])),
              "rows": rows}
    (CT / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in rows[:6]:
        print(f"\nrecall {r['b_recall']} gap {r['gap_ms']}ms  {r['lead']} | {r['b']}\n     {r['hyp']}")


if __name__ == "__main__":
    main()
