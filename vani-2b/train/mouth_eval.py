#!/usr/bin/env python3
"""Evaluate the trained mouth adapter on held-out clauses."""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache

sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from mouth_asr import norm, wer

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
OUT = ROOT / "eval"
SR = 24_000
LIMIT = 40
MAX_FRAMES = 200


@torch.no_grad()
def synth(model, proc, codec, text: str):
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
        torch.tensor(proc.tokenizer.encode(text, add_special_tokens=False)),
        torch.tensor([model.config.text_end_token_id]),
    ]).to(DEVICE)[None]
    out = model.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
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
            inputs_embeds=e,
            attention_mask=mask,
            position_ids=torch.tensor([[cache.get_seq_length()]], device=DEVICE),
            past_key_values=cache,
            use_cache=True,
        )
        hidden, cache = step.last_hidden_state[:, -1], step.past_key_values

    frames, eos_at = [], None
    for i in range(MAX_FRAMES):
        history = torch.stack(frames[-64:]) if frames else None
        codes = model.sample_audio_frame(hidden[0], temperature=0.4, repetition_penalty=1.2, history=history)
        if bool((codes == model.config.audio_eos_code).any()):
            eos_at = i
            break
        frames.append(codes)
        feed(model.model.get_audio_out_embeds(codes[None]))
    codes = torch.stack(frames).t()
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    return wav, len(frames), eos_at


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in json.load(open(ROOT / "train_pairs.json")) if p["split"] == "holdout"][:LIMIT]
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "meta.jsonl") if '"error"' not in l}

    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload()
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()

    rows = []
    for i, p in enumerate(pairs, 1):
        wav, frames, eos_at = synth(model, proc, codec, p["text"])
        sf.write(OUT / f"{p['id']}.wav", wav, SR)
        rows.append({"id": p["id"], "text": p["text"], "frames": frames, "eos_at": eos_at,
                     "audio_s": round(len(wav) / SR, 2),
                     "teacher_frames": meta[p["id"]]["frames"],
                     "words_per_s": round(len(p["text"].split()) / max(len(wav) / SR, 1e-9), 2)})
        if i % 10 == 0:
            print(f"synth {i}/{len(pairs)}", flush=True)

    del model, base
    torch.cuda.empty_cache()

    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    for i, r in enumerate(rows, 1):
        hyp = t.transcribe(OUT / f"{r['id']}.wav").text
        r["hyp"] = hyp
        r["wer"] = round(wer(norm(r["text"]), norm(hyp)), 4)
        if i % 10 == 0:
            print(f"asr {i}/{len(rows)}", flush=True)

    import statistics as st
    report = {
        "n": len(rows),
        "exact": round(sum(r["wer"] == 0 for r in rows) / len(rows), 4),
        "mean_wer": round(st.mean(r["wer"] for r in rows), 4),
        "median_words_per_s": round(st.median(r["words_per_s"] for r in rows), 2),
        "dropped_tail": round(sum(r["eos_at"] is None or r["frames"] < 0.7 * r["teacher_frames"] for r in rows) / len(rows), 4),
        "median_frame_ratio": round(st.median(r["frames"] / r["teacher_frames"] for r in rows), 3),
        "rows": rows,
    }
    (ROOT / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in rows[:5]:
        print(f"\n{r['wer']}  {r['text']}\n     {r['hyp']}")


if __name__ == "__main__":
    main()
