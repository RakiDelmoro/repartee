#!/usr/bin/env python3
"""Four-check set for the joint pilot: duplex, injection, and voice."""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, DynamicCache

sys.path.insert(0, ".")
sys.path.insert(0, "vani-2b/train")
from inject_probe import DEVICE
from continuity_eval import b_ratio, gap_ms, synth as ct_synth
from mouth_asr import norm

MODEL_ID = "models/LFM3-Omni-2B-hf"
ROOT = Path("data/probe/mouth")
CT = ROOT / "continuity"
JOINT = Path("data/probe/duplex/joint")
OUT = Path("data/probe/samples/joint_pilot")
SR = 24_000
CONV = Path("data/open-yap/pilot/conversations/conv_c86652a21050")
START = 30.0
T = 63


@torch.no_grad()
def duplex_sample(model, proc, codec, mode_head, start_s=START, t_frames=T):
    u, _ = librosa.load(CONV / "speaker_b.flac", sr=16_000, mono=True, offset=start_s, duration=t_frames * 0.08)
    sf.write("/tmp/joint_u.wav", u, 16_000)
    inp = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": "/tmp/joint_u.wav"}]}],
        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
    )
    ue, _ = model.model.get_audio_features_stream(inp["input_features"].to(DEVICE), None, True)
    ue = ue[0, :t_frames]
    n = ue.shape[0]
    hop = int(0.08 * 16_000)
    m = len(u) // hop
    rms = np.sqrt((u[:m * hop].reshape(m, hop) ** 2).mean(1))
    user_active = (rms > 0.01)[:n]
    silence = codec.encode(torch.zeros(1, 1920, device=DEVICE)).audio_codes[0, :, 0]
    embed = model.get_input_embeddings()
    listen_id = proc.tokenizer.encode(" listen", add_special_tokens=False)[0]
    listen = embed(torch.tensor([listen_id], device=DEVICE))[0]
    seq = [embed(torch.tensor([[model.config.text_config.bos_token_id]], device=DEVICE))[0, 0]]
    frames, modes = [], []
    run = 0
    for t in range(n):
        seq.append(ue[t])
        x = torch.stack(seq)[None]
        out = model.model.language_model(
            inputs_embeds=x,
            attention_mask=torch.ones((1, len(seq)), dtype=torch.long, device=DEVICE),
            use_cache=False,
        )
        h = out.last_hidden_state[:, -1]
        run = 0 if user_active[t] else run + 1
        sil = min(run, 25) / 25.0
        feats = torch.tensor([[float(user_active[t]), sil, 1.0 - float(user_active[t])]], device=DEVICE)
        mode = int(mode_head(torch.cat([h.float(), feats], -1)).argmax(-1))
        if mode == 1:
            history = torch.stack(frames[-64:]) if frames else None
            codes = model.sample_audio_frame(h[0], temperature=0.0, repetition_penalty=1.5, history=history)
            if bool((codes == model.config.audio_eos_code).any()):
                seq.append(listen); frames.append(silence); modes.append(0)
            else:
                seq.append(model.model.get_audio_out_embeds(codes[None])[0])
                frames.append(codes)
                modes.append(1)
        else:
            seq.append(listen)
            frames.append(silence)
            modes.append(0)
    codes = torch.stack(frames).t()
    wav = codec.decode(codes[None]).audio_values[0].float().cpu().numpy().squeeze()
    return wav, modes, start_s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    codec = proc.audio_tokenizer.to(device=DEVICE, dtype=torch.float32).eval()
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, trust_remote_code=True
    ).to(DEVICE).eval()
    mouth = PeftModel.from_pretrained(base, ROOT / "lora").merge_and_unload().eval()
    sw = PeftModel.from_pretrained(mouth, ROOT / "switch" / "lora").merge_and_unload().eval()
    cont = PeftModel.from_pretrained(sw, CT / "lora_v2").merge_and_unload().eval()
    model = PeftModel.from_pretrained(cont, JOINT / "lora").merge_and_unload().eval()
    mode_head = nn.Linear(model.config.text_config.hidden_size + 3, 2).to(DEVICE)
    mode_head.load_state_dict(torch.load(JOINT / "mode_head.pt", map_location=DEVICE))
    mode_head.eval()

    wav, modes, start = duplex_sample(model, proc, codec, mode_head)
    sf.write(OUT / "duplex.wav", wav, SR)
    mode_info = {"start_s": start, "frames": len(modes), "speak_frames": int(sum(modes)),
                 "listen_frames": int(len(modes) - sum(modes))}

    lab = np.load(Path("data/open-yap/pilot/labels") / f"{CONV.name}.npz")
    a_s, b_s = lab["a_speak"], lab["b_speak"]
    cands = [s0 for s0 in range(0, len(a_s) - T, 10)
             if b_s[s0:s0 + T].sum() < 5 and a_s[s0:s0 + T].any()]
    best = max(cands, key=lambda s0: int(a_s[s0:s0 + T].sum())) if cands else 0
    wav2, modes2, start2 = duplex_sample(model, proc, codec, mode_head, start_s=best * 0.08)
    sf.write(OUT / "duplex_agentturn.wav", wav2, SR)
    agent_info = {"start_s": round(best * 0.08, 2), "frames": len(modes2),
                  "speak_frames": int(sum(modes2)), "listen_frames": int(len(modes2) - sum(modes2))}

    meta = [m for m in json.load(open(CT / "meta.json")) if m["split"] == "holdout"][:2]
    inj = []
    for i, m in enumerate(meta, 1):
        w = ct_synth(model, proc, codec, m["lead_text"], m["b_text"], m["cut_frames"])
        path = OUT / f"injection_{i}.wav"
        sf.write(path, w, SR)
        inj.append({"lead": m["lead_text"], "answer": m["b_text"], "path": path.name,
                    "s": round(len(w) / SR, 2), "gap_ms": gap_ms(path, m["cut_s"])})

    del model, cont, mouth, base
    torch.cuda.empty_cache()
    from app.asr import QwenTranscriber
    t = QwenTranscriber()
    report = {"duplex_user": mode_info, "duplex_user_asr": t.transcribe(OUT / "duplex.wav").text,
              "duplex_agent": agent_info, "duplex_agent_asr": t.transcribe(OUT / "duplex_agentturn.wav").text,
              "injection": inj}
    for r in inj:
        r["asr"] = t.transcribe(OUT / r["path"]).text
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
