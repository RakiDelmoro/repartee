#!/usr/bin/env python3
"""Evaluate the switch adapter against the mouth-only model on held-out pairs."""

import json
import sys
from pathlib import Path

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
SW = ROOT / "switch"
OUT = SW / "eval"
SR = 24_000
LIMIT = 12
MAX_FRAMES = 220


@torch.no_grad()
def synth(model, proc, codec, prefix: str, b_text: str, switch_frames: int):
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
        torch.tensor(proc.tokenizer.encode(prefix, add_special_tokens=False)),
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

    for i in range(MAX_FRAMES):
        if i == switch_frames and not switched:
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


def b_subsequence(hyp: str, b_text: str) -> tuple[float, bool]:
    h, b = norm(hyp), norm(b_text)
    if not b:
        return 0.0, False
    # longest common subsequence ratio of B in the hypothesis
    prev = [0] * (len(h) + 1)
    for w in b:
        cur = [0]
        for j, x in enumerate(h, 1):
            cur.append(prev[j - 1] + 1 if w == x else max(prev[j], cur[j - 1]))
        prev = cur
    ratio = prev[-1] / len(b)
    return ratio, ratio == 1.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in (SW / "pairs.jsonl").read_text().splitlines() if '"split"' in l]
    pairs = [p for p in pairs if p["split"] == "holdout"][:LIMIT]

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()

    for i, p in enumerate(pairs, 1):
        wav = synth(mouth, proc, codec, p["prefix_text"], p["b_text"], p["switch_frames"])
        sf.write(OUT / f"{p['id']}_mouth.wav", wav, SR)
        print(f"mouth {i}/{len(pairs)}", flush=True)

    switch = PeftModel.from_pretrained(mouth, SW / "lora").merge_and_unload().eval()
    for i, p in enumerate(pairs, 1):
        wav = synth(switch, proc, codec, p["prefix_text"], p["b_text"], p["switch_frames"])
        sf.write(OUT / f"{p['id']}_switch.wav", wav, SR)
        print(f"switch {i}/{len(pairs)}", flush=True)

    del switch, mouth, base
    torch.cuda.empty_cache()

    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    rows = []
    for p in pairs:
        rec = {"id": p["id"], "prefix": p["prefix_text"], "b": p["b_text"],
               "switch_s": round(p["switch_frames"] * 0.08, 2)}
        for kind in ("mouth", "switch"):
            path = OUT / f"{p['id']}_{kind}.wav"
            hyp = t.transcribe(path).text
            ratio, present = b_subsequence(hyp, p["b_text"])
            rec[kind] = {"hyp": hyp, "s": round(sf.info(path).duration, 2),
                         "b_recall": round(ratio, 3), "b_present": present}
        rows.append(rec)
        print(rec["id"], "| mouth B", rec["mouth"]["b_present"], "| switch B", rec["switch"]["b_present"], flush=True)

    report = {
        "n": len(rows),
        "mouth_b_present": round(sum(r["mouth"]["b_present"] for r in rows) / len(rows), 3),
        "switch_b_present": round(sum(r["switch"]["b_present"] for r in rows) / len(rows), 3),
        "mouth_b_recall": round(sum(r["mouth"]["b_recall"] for r in rows) / len(rows), 3),
        "switch_b_recall": round(sum(r["switch"]["b_recall"] for r in rows) / len(rows), 3),
        "rows": rows,
    }
    (SW / "switch_eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in rows[:5]:
        print(f"\n{r['id']} A={r['prefix']!r} B={r['b']!r}")
        print("  mouth :", r["mouth"]["hyp"])
        print("  switch:", r["switch"]["hyp"])


if __name__ == "__main__":
    main()
