REPARTEE — Build Blueprint v1.0                                                                                                             
                                                                                                                                             
 Project: Repartee                                                                                                                           
 Model: repartee-2b                                                                                                                          
 Tagline: A 2B conversation model that talks back.                                                                                           
 Status: ready for execution · Date: 2026-10-29                                                                                              
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 0. Instructions to the executing agent                                                                                                      
                                                                                                                                             
 You are the engineer implementing this. Follow the phases in order. Do not skip a gate. Do not substitute a different base model, method,   
 or dataset without recording the reason and getting approval.                                                                               
                                                                                                                                             
 - Every gate has a numeric pass condition. Write the measured value to REPORT.md.                                                           
 - If a gate fails: stop and report. Do not silently continue or lower the threshold.                                                        
 - Never invent an API. Inspect config.json, the model card, and one working forward pass before writing code.                               
 - Never report curated demo clips as evidence. Only gate numbers and the pilot.                                                             
 - One command must rebuild data, one must train, one must serve.                                                                            
                                                                                                                                             
 If a fact cannot be verified from the checkpoint or a cited paper, mark it UNVERIFIED and stop at that gate.                                
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 1. Mission                                                                                                                                  
                                                                                                                                             
 Convert a pretrained 2–3B omni model into a full-duplex speech-to-speech model that:                                                        
                                                                                                                                             
 1. listens and speaks simultaneously — no VAD, no turn detector;                                                                            
 2. supports barge-in and backchannels;                                                                                                      
 3. runs offline on a consumer GPU (≤8 GB int4, ≥1× realtime);                                                                               
 4. keeps ≥98% of the base model's text benchmark scores.                                                                                    
                                                                                                                                             
 Method: add 4 control tokens + frame-level supervision. No new architecture, no from-scratch pretraining, no codec training.                
                                                                                                                                             
 Non-goals: beating a 9B model on knowledge; multilingual beyond the base; video; tool calling (phase 2, out of scope).                      
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 2. Naming                                                                                                                                   
                                                                                                                                             
 ┌───────────────────────────┬───────────────────┐                                                                                           
 │ Thing                     │ Name              │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Project                   │ Repartee          │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Model (duplex)            │ repartee-2b       │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Base (stages 1–2, no DPO) │ repartee-2b-base  │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Chat (after DPO)          │ repartee-2b-chat  │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Data mix                  │ repartee-mix      │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Serving runtime           │ repartee-serve    │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Eval harness              │ repartee-bench    │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ Repo                      │ repartee          │                                                                                           
 ├───────────────────────────┼───────────────────┤                                                                                           
 │ HF namespace              │ <org>/repartee-2b │                                                                                           
 └───────────────────────────┴───────────────────┘                                                                                           
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 3. Fixed decisions                                                                                                                          
                                                                                                                                             
 ┌───────────────┬───────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┐ 
 │ Decision      │ Value                                                         │ Fallback                                                │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Base model    │ LiquidAI/LFM3-Omni-2B (2.17B, causal audio encoder,           │ Qwen/Qwen2.5-Omni-3B + D2 encoder distillation (arXiv   │ 
 │               │ LiquidCodec)                                                  │ 2610.17124)                                             │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Method        │ Recipe 3: special tokens + SFT + DPO (BayLing-style, arXiv    │ —                                                       │ 
 │               │ 2606.14528)                                                   │                                                         │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Frame rate    │ 12.5 Hz (80 ms)                                               │ —                                                       │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Control       │ <listen>, <speak>, <backchannel>, <stop>                      │ —                                                       │ 
 │ tokens        │                                                               │                                                         │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Real data     │ Open Yap 1K (1,000 h) + otoSpeech 280 h + turn-104 h          │ CANDOR (research only)                                  │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Synthetic     │ 8,000 h                                                       │ —                                                       │ 
 │ data          │                                                               │                                                         │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Training      │ 8×A100 80 GB, 1–2 days                                        │ 8×H100, 4–8 h                                           │ 
 ├───────────────┼───────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤ 
 │ Serving       │ 80 ms loop, KV cache, AEC3                                    │ —                                                       │ 
 └───────────────┴───────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────┘ 
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 4. Gate 0 — verify before writing code (blocking)                                                                                           
                                                                                                                                             
 Run against the actual checkpoint. Record results in REPORT.md §Gate0.                                                                      
                                                                                                                                             
 ┌─────┬───────────────────┬───────────────────────────────────────────────────────────────────────────────────┬───────────────────────────┐ 
 │ #   │ Check             │ Pass condition                                                                    │ If fail                   │ 
 ├─────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
 │ 0.1 │ License           │ LFM Open License permits your deployment                                          │ switch to fallback base   │ 
 ├─────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
 │ 0.2 │ Encoder causality │ 1 state / 80 ms; output at t unchanged when audio after t is zeroed               │ distill (fallback)        │ 
 ├─────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
 │ 0.3 │ Codec streaming   │ decoder emits ≤160 ms chunks, stable over 30 s                                    │ decouple TTS              │ 
 ├─────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
 │ 0.4 │ Interruptibility  │ forcing EOS stops audio within 2 frames                                           │ add explicit <stop> first │ 
 ├─────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
 │ 0.5 │ Head layout       │ identify text head, audio head/depthformer, connector module names in config.json │ —                         │ 
 └─────┴───────────────────┴───────────────────────────────────────────────────────────────────────────────────┴───────────────────────────┘ 
                                                                                                                                             
 Stop and report if 0.2 or 0.3 fail — the plan changes to the fallback.                                                                      
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 5. Repo layout                                                                                                                              
                                                                                                                                             
 ```                                                                                                                                         
repartee/
  data/
    manifests/          # one JSONL per source
    raw/                # downloads (gitignored)
    tokens/             # tokens_00000.bin + index.json
  src/
    schema.py           # dataclasses + control-label derivation
    build_real.py       # Open Yap / otoSpeech -> manifest
    build_synth.py      # LLM -> TTS -> manifest
    tokenize.py         # manifest -> 12.5 Hz shards
    dataset.py          # torch Dataset over shards
    model.py            # load base, add tokens, connector
    train_stage1.py
    train_stage2.py
    train_dpo.py
  eval/
    fdb.py              # Full-Duplex-Bench runner
    text_regression.py  # MMLU / GSM8K subset
    pilot.py            # 20-scenario live test
  serve/
    loop.py             # 80 ms scheduler
    aec.py
  REPORT.md
 ```                                                                                                                                         
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 6. Data schema                                                                                                                              
                                                                                                                                             
 One JSONL record per conversation:                                                                                                          
                                                                                                                                             
 ```json                                                                                                                                     
{
  "id": "oy1k_000123",
  "source": "open_yap_1k",
  "license": "commercial",
  "duration_s": 1830.4,
  "channels": {
    "user":  "data/raw/oy1k/000123_a.flac",
    "agent": "data/raw/oy1k/000123_b.flac"
  },
  "turns": [{"speaker": "user", "start": 0.0, "end": 1.24, "text": "..."}]
}
 ```                                                                                                                                         
                                                                                                                                             
 Control labels at 12.5 Hz (schema.py, unit-tested):                                                                                         
                                                                                                                                             
 ┌──────────────────────────────────────────────────────┬─────────────┐                                                                      
 │ Frame condition                                      │ Label       │                                                                      
 ├──────────────────────────────────────────────────────┼─────────────┤                                                                      
 │ agent silent, user speaking                          │ LISTEN      │                                                                      
 ├──────────────────────────────────────────────────────┼─────────────┤                                                                      
 │ agent speech onset                                   │ SPEAK       │                                                                      
 ├──────────────────────────────────────────────────────┼─────────────┤                                                                      
 │ agent utterance < 0.8 s and overlaps user            │ BACKCHANNEL │                                                                      
 ├──────────────────────────────────────────────────────┼─────────────┤                                                                      
 │ user speech onset during agent speech, within 200 ms │ STOP        │                                                                      
 ├──────────────────────────────────────────────────────┼─────────────┤                                                                      
 │ otherwise                                            │ PAD         │                                                                      
 └──────────────────────────────────────────────────────┴─────────────┘                                                                      
                                                                                                                                             
 Audio: user resampled to 16 kHz mono (encoder input), agent to 24 kHz mono (codec target), both mapped to the same 80 ms grid.              
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 7. Phase 1 — data build                                                                                                                     
                                                                                                                                             
 ### 7.1 Real (~1,300 h)                                                                                                                     
                                                                                                                                             
 1. hf download TheAgenticDataCompany/open-yap-1k (request access; sample is public). 1,000 h, 48 kHz, dual-channel, commercial use.         
 2. hf download otoearth/otoSpeech-full-duplex-280h and otoearth/otoSpeech-full-duplex-turn-104h (gated, CC-BY-4.0; accept terms).           
 3. Normalize to schema; drop conversations with >20% clipping or a missing channel.                                                         
 4. Hold out 50 speakers and 100 conversations. Never train on them.                                                                         
                                                                                                                                             
 ### 7.2 Synthetic (8,000 h)                                                                                                                 
                                                                                                                                             
 1. LLM writes 2–8 turn dialogues: 70% assistant/task, 20% casual, 10% backchannel-heavy. Fixed agent persona and fixed agent voice id.      
 2. Render with a commercially licensed TTS: Dia (Apache-2.0), Chatterbox (MIT), CosyVoice2 (Apache-2.0), Kokoro (Apache-2.0). Never         
    XTTS-v2, F5-TTS, or Fish weights.                                                                                                        
 3. Compose on one timeline, exact rates:                                                                                                    
   - inter-turn gap 0.64 s (80% of turns)                                                                                                    
   - barge-in: cut agent turn, pad silence (20% of agent turns)                                                                              
   - backchannel during user speech (10% of user turns)                                                                                      
   - 100–400 ms overlap at 15% of turn boundaries                                                                                            
 4. User-channel augmentation: gain −24…+15 dB (50%), noise −30…+6 dB SNR (30%), reverb (30%), agent echo 0–20% gain / 100–500 ms delay      
    (30%).                                                                                                                                   
 5. Reject synthetic files whose agent voice similarity drifts (ECAPA cosine < 0.8 vs reference).                                            
                                                                                                                                             
 Acceptance: ≥1,300 h real, ≥8,000 h synthetic, scripted license audit passes.                                                               
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 8. Phase 2 — tokenize once                                                                                                                  
                                                                                                                                             
 tokenize.py → data/tokens/.                                                                                                                 
                                                                                                                                             
 - Both channels at 12.5 Hz, uint16, layout per conversation: user[T,8] | agent[T,8] | text[T] | control[T].                                 
 - Target: 15–30 GB total. Do not store continuous features (≈1 TB).                                                                         
 - Verify: decode 10 random shards, resynthesize, WER < 5% vs transcript, UTMOS within 0.1 of source.                                        
                                                                                                                                             
 Acceptance: dataloader never touches FLAC during training; random batch load < 5 ms.                                                        
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 9. Phase 3 — model surgery                                                                                                                  
                                                                                                                                             
 model.py:                                                                                                                                   
                                                                                                                                             
 - Load base in bf16. Add 4 tokens; initialize their embeddings from the mean of existing embeddings (not random).                           
 - Add/verify the audio connector (encoder → LLM) as the only new module. Log its parameter count (target 5–30M).                            
 - Freeze: audio encoder, LiquidCodec, depthformer, vision encoder, all norms.                                                               
 - Trainable: connector, 4 token embeddings; later the LLM; later LoRA.                                                                      
                                                                                                                                             
 Assert: stage-1 trainable params < 2% of total. Print the per-module freeze table into REPORT.md.                                           
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 10. Phase 4 — training                                                                                                                      
                                                                                                                                             
 ### Stage 1 — adapter warm-up                                                                                                               
                                                                                                                                             
 ┌───────────┬──────────────────────────────────────────────┐                                                                                
 │ Setting   │ Value                                        │                                                                                
 ├───────────┼──────────────────────────────────────────────┤                                                                                
 │ Trainable │ connector + 4 embeddings                     │                                                                                
 ├───────────┼──────────────────────────────────────────────┤                                                                                
 │ Data      │ 200 h mix, seq 4096 frames (~5.5 min)        │                                                                                
 ├───────────┼──────────────────────────────────────────────┤                                                                                
 │ Batch     │ 8 per GPU × 8 GPUs × grad-accum 2 = 128 seqs │                                                                                
 ├───────────┼──────────────────────────────────────────────┤                                                                                
 │ LR        │ 1e-3, cosine, 500 warmup                     │                                                                                
 ├───────────┼──────────────────────────────────────────────┤                                                                                
 │ Steps     │ 3,000–6,000                                  │                                                                                
 ├───────────┼──────────────────────────────────────────────┤                                                                                
 │ Precision │ bf16 + activation checkpointing              │                                                                                
 └───────────┴──────────────────────────────────────────────┘                                                                                
                                                                                                                                             
 Gate 4: text-only MMLU/GSM8K within 2 points of the Gate 0 baseline. Otherwise stop.                                                        
                                                                                                                                             
 ### Stage 2 — duplex SFT                                                                                                                    
                                                                                                                                             
 ┌───────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐              
 │ Setting   │ Value                                                                                                          │              
 ├───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤              
 │ Trainable │ LLM at 5e-5 + connector at 1e-3 + tokens                                                                       │              
 ├───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤              
 │ Frozen    │ encoder, codec, depthformer                                                                                    │              
 ├───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤              
 │ Data      │ full 9,300 h, 2 epochs                                                                                         │              
 ├───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤              
 │ Loss      │ 3·CE_text + 2·CE_control + 1·CE_audio(optional); control weights <speak>/<stop> ×8, <backchannel> ×4, PAD ×0.3 │              
 ├───────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤              
 │ Steps     │ 800–2,000 per epoch; measure and log throughput                                                                │              
 └───────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘              
                                                                                                                                             
 Gate 5: FDB v1 — pause TOR < 60%, interruption success > 90%, TTFA < 400 ms, backchannel resume > 70%.                                      
                                                                                                                                             
 ### Stage 3 — DPO on timing                                                                                                                 
                                                                                                                                             
 - 5,000 pairs: same context, two timings (e.g. 200 ms vs 1,200 ms onset; barge-in vs talk-over), labelled by FDB metrics.                   
 - LoRA r=16 on LLM, β=0.1, LR 5e-6, 1,000–2,000 steps.                                                                                      
 - Gate 6: no regression on Gates 4–5.                                                                                                       
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 11. Phase 5 — serving                                                                                                                       
                                                                                                                                             
 serve/loop.py, fixed 80 ms tick:                                                                                                            
                                                                                                                                             
 ```                                                                                                                                         
tick:
  read 80 ms mic
  -> AEC3 (reference = last 500 ms played)
  -> encoder
  build frame input (user + previous agent 80 ms + previous text)
  LLM forward with KV cache -> text / control token
  if <speak>: start TTS
  if <stop>:  drop queued audio, keep 80 ms tail
  if <backchannel>: enqueue short clip
  push 80 ms decoded audio to jitter buffer (20-40 ms)
 ```                                                                                                                                         
                                                                                                                                             
 Requirements: single writer; no buffer growth over 10 min; RTF < 0.5 on RTX 4090 int4; TTFA ≤ 350 ms on LAN; barge-in stop ≤ 2 frames.      
                                                                                                                                             
 Gate 7: 10-minute live session, zero self-interruptions, flat memory, documented p50/p95.                                                   
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 12. Phase 6 — evaluation                                                                                                                    
                                                                                                                                             
 ┌───────────────────┬────────────────────────────────────────────────────┐                                                                  
 │ Test              │ Pass                                               │                                                                  
 ├───────────────────┼────────────────────────────────────────────────────┤                                                                  
 │ FDB v1 + v1.5     │ numbers recorded, no regression vs Gate 5          │                                                                  
 ├───────────────────┼────────────────────────────────────────────────────┤                                                                  
 │ Text regression   │ within 2 points of base                            │                                                                  
 ├───────────────────┼────────────────────────────────────────────────────┤                                                                  
 │ 20-scenario pilot │ corrections, interruptions, noise, names, refusals │                                                                  
 ├───────────────────┼────────────────────────────────────────────────────┤                                                                  
 │ Consumer run      │ int4 ≤ 8 GB, ≥1× realtime on RTX 4090 and M4 Max   │                                                                  
 └───────────────────┴────────────────────────────────────────────────────┘                                                                  
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 13. Definition of Done                                                                                                                      
                                                                                                                                             
 - REPORT.md with Gate 0–7 numbers and the freeze table                                                                                      
 - repartee-2b weights + tokenizer + serving code                                                                                            
 - int4 ≤ 8 GB; runs on RTX 4090 and M4 Max                                                                                                  
 - FDB v1 + v1.5 numbers; text regression within 2 points                                                                                    
 - 20-scenario pilot executed and scored                                                                                                     
 - License audit: every dataset and TTS listed with terms                                                                                    
 - One command each: rebuild data, train, serve                                                                                              
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 14. Stop conditions                                                                                                                         
                                                                                                                                             
 ┌──────────────────────────────────┬──────────────────────────────────────────────────────┐                                                 
 │ Symptom                          │ Action                                               │                                                 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┤                                                 
 │ Gate 4 fails after 3 LR restarts │ freeze more, check adapter; do not proceed           │                                                 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┤                                                 
 │ Pause TOR > 80% after 1,000 h    │ data problem: add silence/backchannel distribution   │                                                 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┤                                                 
 │ Text benchmarks drop > 3 points  │ stop; mix text-only batches; lower LR                │                                                 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┤                                                 
 │ Self-interruption in live test   │ echo augmentation missing; fix data, retrain stage 2 │                                                 
 ├──────────────────────────────────┼──────────────────────────────────────────────────────┤                                                 
 │ Codec output unstable            │ decouple TTS; do not train the codec                 │                                                 
 └──────────────────────────────────┴──────────────────────────────────────────────────────┘                                                 
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 15. Never do                                                                                                                                
                                                                                                                                             
 1. Never train the codec or the encoder from scratch.                                                                                       
 2. Never pretrain the LLM on audio-only data for this goal.                                                                                 
 3. Never use CANDOR, Fisher, or podcasts commercially without checking terms.                                                               
 4. Never train on diarized mono and call it dual-channel.                                                                                   
 5. Never start stage 2 before Gate 4 passes.                                                                                                
 6. Never report demo clips as evidence.                                                                                                     
 7. Never exceed 2% trainable params at stage 1.                                                                                             
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 16. Schedule                                                                                                                                
                                                                                                                                             
 ┌──────┬────────────────────────────────┬─────────────────────────────┐                                                                     
 │ Week │ Work                           │ Exit                        │                                                                     
 ├──────┼────────────────────────────────┼─────────────────────────────┤                                                                     
 │ 1    │ Gate 0, schema, 100 h pilot    │ Gate 0 report               │                                                                     
 ├──────┼────────────────────────────────┼─────────────────────────────┤                                                                     
 │ 2    │ real data + synthetic pipeline │ 1,300 h real, 1,000 h synth │                                                                     
 ├──────┼────────────────────────────────┼─────────────────────────────┤                                                                     
 │ 3    │ tokenize, stage 1              │ Gate 4                      │                                                                     
 ├──────┼────────────────────────────────┼─────────────────────────────┤                                                                     
 │ 4    │ stage 2 on 5k h                │ Gate 5 draft                │                                                                     
 ├──────┼────────────────────────────────┼─────────────────────────────┤                                                                     
 │ 5    │ scale to 9.3k h, stage 3       │ Gates 5–6                   │                                                                     
 ├──────┼────────────────────────────────┼─────────────────────────────┤                                                                     
 │ 6    │ serving, AEC, pilot            │ Gates 7, DoD                │                                                                     
 └──────┴────────────────────────────────┴─────────────────────────────┘                                                                     
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 17. Report format (use verbatim)                                                                                                            
                                                                                                                                             
 ```markdown                                                                                                                                 
## Gate N — <name>
- Command:
- Measured:
- Threshold:
- Result: PASS | FAIL
- Artifacts:
- Notes / UNVERIFIED:
 ```                                                                                                                                         
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 18. Glossary (for the human reviewer)                                                                                                       
                                                                                                                                             
 ┌───────────────┬────────────────────────────────────────────────────┐                                                                      
 │ Term          │ Meaning                                            │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Full-duplex   │ both sides can speak at once (phone call)          │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Half-duplex   │ one at a time (walkie-talkie)                      │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Encoder       │ ears: waveform → features                          │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Codec         │ audio compressor / decompressor                    │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Depthformer   │ predicts audio codes per frame (the "mouth")       │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Connector     │ adapter from encoder space to LLM space            │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Control token │ special word meaning listen/speak/backchannel/stop │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ 80 ms frame   │ the heartbeat; 12.5 decisions per second           │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Barge-in      │ user interrupts the model                          │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Backchannel   │ "mm-hm" that doesn't take the turn                 │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ Freeze        │ weights don't update                               │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ LoRA          │ small trainable matrices on a frozen base          │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ SFT           │ practice with example conversations                │                                                                      
 ├───────────────┼────────────────────────────────────────────────────┤                                                                      
 │ DPO           │ "this timing was better than that one" training    │                                                                      
 └───────────────┴────────────────────────────────────────────────────┘                                                                      
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            
                                                                                                                                             
 19. References                                                                                                                              
                                                                                                                                             
 - LFM3-Omni-2B — https://huggingface.co/LiquidAI/LFM3-Omni-2B                                                                               
 - BayLing-Duplex (special tokens + SFT + DPO) — arXiv 2606.14528                                                                            
 - SALM-Duplex (duplex without speech pretraining) — arXiv 2505.15670                                                                        
 - D2 (half-duplex → full-duplex conversion) — arXiv 2610.17124                                                                              
 - NemotronLabs VoiceChat (frame-level BOS/EOS/padding supervision) — arXiv 2609.21967                                                       
 - Moshi (dual-stream codec reference) — arXiv 2410.00037                                                                                    
 - Full-Duplex-Bench — arXiv 2503.04721                                                                                                      
 - Open Yap 1K — https://theagenticdatacompany.com/open-yap-1k                                                                               
 - otoSpeech — https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h                                                           
                                                                                                                                             
 ────────────────────────────────────────────────────────────────────────────────                                                            