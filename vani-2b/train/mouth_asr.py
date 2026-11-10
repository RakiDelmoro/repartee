#!/usr/bin/env python3
"""Verify the TTS teacher: exact-match rate and speaking rate on the mouth corpus."""

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path("data/probe/mouth")
CACHE = ROOT / "asr_cache.jsonl"

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ((" " + ONES[n % 10]) if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + ((" " + int_words(n % 100)) if n % 100 else "")
    if n < 1_000_000:
        return int_words(n // 1000) + " thousand" + ((" " + int_words(n % 1000)) if n % 1000 else "")
    return str(n)


def expand_number(tok: str) -> str:
    tok = tok.replace(",", "")
    if "." in tok:
        a, b = tok.split(".", 1)
        left = int_words(int(a)) if a else "zero"
        digits = " ".join(ONES[int(d)] for d in b)
        return f"{left} point {digits}"
    try:
        return int_words(int(tok))
    except ValueError:
        return tok


ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
            7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
            13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
            17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
            21: "twenty first", 22: "twenty second", 23: "twenty third", 24: "twenty fourth",
            25: "twenty fifth", 26: "twenty sixth", 27: "twenty seventh", 28: "twenty eighth",
            29: "twenty ninth", 30: "thirtieth", 31: "thirty first"}
CONTRACTIONS = {
    "we're": "we are", "we've": "we have", "we'll": "we will", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "it's": "it is", "that's": "that is",
    "you're": "you are", "you've": "you have", "they're": "they are",
    "they've": "they have", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "won't": "will not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
}


def norm(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ORDINALS.get(int(m.group(1)), int_words(int(m.group(1))) + "th"), text)
    text = text.replace("—", " ").replace("-", " ")
    def money(currency):
        def repl(m):
            num, scale = m.group(1), (m.group(2) or "")
            return f" {num} {scale} {currency} "
        return repl

    text = re.sub(r"\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?", money("dollars"), text)
    text = re.sub(r"£\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?", money("pounds"), text)
    text = re.sub(r"€\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?", money("euros"), text)
    text = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", lambda m: " " + expand_number(m.group(0)) + " ", text)
    text = re.sub(r"[^a-z' ]+", " ", text)
    return text.split()


def wer(ref: list[str], hyp: list[str]) -> float:
    prev = list(range(len(hyp) + 1))
    for i, w in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (w != h)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def main() -> None:
    corpus = {json.loads(line)["id"]: json.loads(line) for line in (ROOT / "corpus.jsonl").read_text().splitlines()}
    meta = {}
    if (ROOT / "meta.jsonl").exists():
        for line in (ROOT / "meta.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if "error" not in rec:
                meta[rec["id"]] = rec
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["id"]] = rec["hyp"]

    wavs = sorted((ROOT / "wav").glob("*.wav"))
    todo = [p for p in wavs if p.stem not in cache]
    if todo:
        sys.path.insert(0, ".")
        from app.asr import QwenTranscriber
        t = QwenTranscriber()
        with CACHE.open("a") as f:
            for i, p in enumerate(todo, 1):
                try:
                    hyp = t.transcribe(p).text
                except Exception as exc:
                    hyp = ""
                    print("ERR", p.stem, exc, flush=True)
                cache[p.stem] = hyp
                f.write(json.dumps({"id": p.stem, "hyp": hyp}) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"asr {i}/{len(todo)}", flush=True)

    rows = []
    for pid, ref_text in ((c["id"], c["text"]) for c in corpus.values() if c["id"] in cache):
        r = norm(ref_text)
        h = norm(cache[pid])
        w = wer(r, h)
        rec = {"id": pid, "split": corpus[pid]["split"], "family": corpus[pid]["family"],
               "text": ref_text, "hyp": cache[pid], "wer": round(w, 4),
               "words_per_s": meta.get(pid, {}).get("words_per_s")}
        rows.append(rec)

    def summary(sub):
        if not sub:
            return {}
        ws = [r["wer"] for r in sub]
        rates = [r["words_per_s"] for r in sub if r["words_per_s"]]
        return {
            "n": len(sub),
            "exact": round(sum(w == 0 for w in ws) / len(ws), 4),
            "mean_wer": round(statistics.mean(ws), 4),
            "median_wer": round(statistics.median(ws), 4),
            "median_words_per_s": round(statistics.median(rates), 2) if rates else None,
        }

    report = {
        "overall": summary(rows),
        "by_split": {s: summary([r for r in rows if r["split"] == s]) for s in ("train", "holdout")},
        "by_family": {f: summary([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})},
        "worst": sorted(rows, key=lambda r: -r["wer"])[:10],
    }
    (ROOT / "asr_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "worst"}, indent=2))
    for r in report["worst"][:5]:
        print(f"\nWER {r['wer']}  {r['family']}  target: {r['text']}\n              hyp:    {r['hyp']}")


if __name__ == "__main__":
    main()
