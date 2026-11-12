#!/usr/bin/env python3
"""Targeted pure-audio duplex probe: user-speaking vs agent-turn windows."""

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
from app.asr import QwenTranscriber

MODEL_ID = "models/LFM3-Audio-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
DUP = Path("data/probe/duplex")
AUDIO = DUP / os.environ.get("AUDIO_DIR", "audio_lora")
CACHE = DUP / "cache_turns"
OUT = Path("data/probe/samples/duplex_audio")
SR = 16000
OUT_SR = 24000


def speech_frames(wav: np.ndarray, sr: int = OUT_SR) -> int:
    hop = int(0.08 * sr); n = len(wav) // hop
    return sum(float(np.sqrt(np.mean(wav[i * hop:(i + 1) * hop] ** 2))) > 0.01 for i in range(n))


def pick(limit: int = 1) -> list[Path]:
    user, agent = [], []
    for path in sorted(DUP.glob("cache*/*.npz")):
        d = np.load(path)
        b = int(d["b_speak"].sum()); s = int(d["speak"].sum())
        if b > 30 and s < 5:
            user.append(path)
    for path in sorted(CACHE.glob("*.npz")):
        d = np.load(path)
        if int(d["speak"].sum()) > 20:
            agent.append(path)
    return user[:limit] + agent[:limit]


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
    embed = model.get_input_embeddings()
    bos = embed(torch.tensor([[model.config.text_config.bos_token_id]], device=DEVICE))[0, 0]
    silence = None
    for path in sorted(CACHE.glob("*.npz")):
        d = np.load(path)
        idx = np.where(~d["speak"])[0]
        if len(idx):
            silence = torch.from_numpy(d["codes"][idx[0]]).long().to(DEVICE)
            break
    if silence is None:
        silence = torch.zeros(model.config.num_codebooks, dtype=torch.long, device=DEVICE)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    eos = int(model.config.audio_eos_code)
    if os.environ.get("SUPPRESS_EOS") == "1":
        orig_sample = model._sample

        def masked_sample(logits, temperature, top_k, repetition_penalty=None, history=None,
                          suppress=(), top_p=None, min_p=None, warpers=None):
            return orig_sample(logits, temperature, top_k, repetition_penalty, history,
                               tuple(set(suppress) | {eos}), top_p, min_p, warpers)

        model._sample = masked_sample
    report = []
    for path in pick(limit=1):
        d = np.load(path)
        u = torch.from_numpy(d["u"]).to(DEVICE).half()
        n = len(u); b = int(d["b_speak"].sum()); s = int(d["speak"].sum())
        print("probe", path.stem, "b_speak", b, "agent_speak", s, flush=True)
        with torch.no_grad():
            seq = torch.cat([bos.view(1, -1), u[0].view(1, -1)], 0)[None]
            modes, frames = [], []
            for t in range(n):
                out = model.model.language_model(
                    inputs_embeds=seq,
                    attention_mask=torch.ones((1, seq.shape[1]), dtype=torch.long, device=DEVICE),
                    use_cache=False,
                )
                history = torch.stack(frames[-64:]) if frames else None
                codes = model.sample_audio_frame(
                    out.last_hidden_state[0, -1], temperature=0.0, repetition_penalty=1.5, history=history
                )
                if bool((codes == model.config.audio_eos_code).any()):
                    codes = silence
                modes.append(codes); frames.append(codes)
                a = model.model.get_audio_out_embeds(codes[None])[0]
                seq = torch.cat([seq, a.view(1, 1, -1)], 1)
                if t + 1 < n:
                    seq = torch.cat([seq, u[t + 1].view(1, 1, -1)], 1)
        if modes:
            codes_t = torch.stack(modes).permute(1, 0)[None]
            audio = codec.decode(codes_t).audio_values[0].detach().float().cpu().numpy().reshape(-1)
        else:
            audio = np.zeros(0, np.float32)
        OUT.mkdir(parents=True, exist_ok=True)
        out_path = OUT / f"{path.stem}.wav"
        if len(audio) < 1600:
            audio = np.zeros(1600, dtype=np.float32)
        sf.write(out_path, audio, OUT_SR)
        got = speech_frames(audio)
        print("frames", len(modes), "speech_frames", got, flush=True)
        report.append({"id": path.stem, "b_speak": b, "agent_speak": s, "frames": len(modes),
                       "out_speech_frames": got, "asr": ""})
    del model
    torch.cuda.empty_cache()
    silence = None
    for path in sorted(CACHE.glob("*.npz")):
        d = np.load(path)
        idx = np.where(~d["speak"])[0]
        if len(idx):
            silence = torch.from_numpy(d["codes"][idx[0]]).long().to(DEVICE)
            break
    asr = QwenTranscriber(os.environ.get("REPARTEE_ASR_MODEL", "models/Qwen3-ASR-0.6B-hf"))
    for r in report:
        if r["out_speech_frames"] == 0:
            r["asr"] = "<silence>"
            continue
        r["asr"] = asr.transcribe(OUT / f"{r['id']}.wav").text
        print("asr", r["id"], repr(r["asr"]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
