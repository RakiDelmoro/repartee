#!/usr/bin/env python3
"""Fair pure-audio duplex probe: distributions over user, agent, and boundary windows."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "vani-2b/train")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inject_probe import DEVICE

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
AUDIO = DUP / os.environ.get("AUDIO_DIR", "audio_lora")
CACHE = DUP / "cache_turns"
OUT = Path("data/probe/samples/duplex_multi")
SR = 16000
OUT_SR = 24000
N = int(os.environ.get("N", "10"))
SAVE_WAV = os.environ.get("SAVE_WAV", "0") == "1"


def speech_frames(wav: np.ndarray, sr: int = OUT_SR) -> int:
    hop = int(0.08 * sr); n = len(wav) // hop
    return sum(float(np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2))) > 0.01 for i in range(n))


def categories():
    user, agent, boundary = [], [], []
    for path in sorted(CACHE.glob("*.npz")):
        z = np.load(path)
        b = z["b_speak"].astype(bool); s = z["speak"].astype(bool)
        if b.sum() >= 30 and s.sum() <= 10:
            user.append(path)
        if s.sum() >= 40 and b.sum() <= 10:
            agent.append(path)
        if b[:25].sum() >= 10 and b[25:].sum() <= 3 and s[25:].sum() >= 10:
            boundary.append(path)
    return user, agent, boundary


def take(paths, n=N):
    if os.environ.get("OVERFIT") == "1":
        return paths[:n]
    if len(paths) <= n:
        return paths
    idx = np.linspace(0, len(paths) - 1, n).astype(int)
    return [paths[i] for i in idx]


def main() -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False
    mouth = PeftModel.from_pretrained(model, ROOT / "lora").merge_and_unload()
    sw = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload()
    cont = PeftModel.from_pretrained(sw, CT / "lora_v2").merge_and_unload()
    model = PeftModel.from_pretrained(cont, AUDIO).merge_and_unload().eval()
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    embed = model.get_input_embeddings()
    bos = embed(torch.tensor([[model.config.text_config.bos_token_id]], device=DEVICE))[0, 0]
    eos = int(model.config.audio_eos_code)
    orig_sample = model._sample

    def masked_sample(logits, temperature, top_k, repetition_penalty=None, history=None,
                      suppress=(), top_p=None, min_p=None, warpers=None):
        return orig_sample(logits, temperature, top_k, repetition_penalty, history,
                           tuple(set(suppress) | {eos}), top_p, min_p, warpers)

    model._sample = masked_sample
    silence = None
    for path in sorted(CACHE.glob("*.npz")):
        d = np.load(path)
        idx = np.where(~d["speak"])[0]
        if len(idx):
            silence = torch.from_numpy(d["codes"][idx[0]]).long().to(DEVICE)
            break

    user, agent, boundary = categories()
    report = {}
    for name, paths in (("user", take(user)), ("agent", take(agent)), ("boundary", take(boundary))):
        rows = []
        for path in paths:
            d = np.load(path)
            u = torch.from_numpy(d["u"]).to(DEVICE).half()
            n = len(u)
            with torch.no_grad():
                seq = torch.cat([bos.view(1, -1), u[0].view(1, -1)], 0)[None]
                modes = []
                for t in range(n):
                    out = model.model.language_model(
                        inputs_embeds=seq,
                        attention_mask=torch.ones((1, seq.shape[1]), dtype=torch.long, device=DEVICE),
                        use_cache=False,
                    )
                    history = torch.stack(modes[-64:]) if modes else None
                    codes = model.sample_audio_frame(
                        out.last_hidden_state[0, -1], temperature=0.0, repetition_penalty=1.5, history=history
                    )
                    if bool((codes == eos).any()):
                        codes = silence
                    modes.append(codes)
                    a = model.model.get_audio_out_embeds(codes[None])[0]
                    seq = torch.cat([seq, a.view(1, 1, -1)], 1)
                    if t + 1 < n:
                        seq = torch.cat([seq, u[t + 1].view(1, 1, -1)], 1)
            codes_t = torch.stack(modes).permute(1, 0)[None]
            audio = codec.decode(codes_t).audio_values[0].detach().float().cpu().numpy().reshape(-1)
            got = speech_frames(audio)
            label = int(d["speak"].sum())
            onset = None
            if name == "boundary":
                hop = int(0.08 * OUT_SR); m = len(audio) // hop
                for i in range(n // 2, m):
                    if float(np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2))) > 0.01:
                        onset = i; break
            rows.append({"id": path.stem, "label_speak": label, "out_speak": got,
                         "frames": n, "ratio": round(got / max(n, 1), 3), "onset": onset})
            if SAVE_WAV:
                OUT.mkdir(parents=True, exist_ok=True)
                sf.write(OUT / f"{name}_{path.stem}.wav", audio, OUT_SR)
            print(name, path.stem, "label", label, "out", got, "onset", onset, flush=True)
        ratios = [r["ratio"] for r in rows]
        report[name] = {
            "n": len(rows), "median_ratio": round(float(np.median(ratios)), 3),
            "mean_ratio": round(float(np.mean(ratios)), 3),
            "any_speech_pct": round(100 * sum(r["out_speak"] >= 5 for r in rows) / max(len(rows), 1), 1),
            "rows": rows,
        }
        print(name, report[name]["median_ratio"], report[name]["any_speech_pct"], flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
