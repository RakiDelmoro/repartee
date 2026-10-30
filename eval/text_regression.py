"""Direct MMLU / GSM8K subset regression for Repartee.

This is a small, honest, in-repo text benchmark.  It does **not** import
``lm-eval``: MMLU is scored by summed log-likelihood over the four answer
letters, and GSM8K by greedy generation plus exact match of the final numeric
answer.  The point is a deterministic before/after check, not a leaderboard
number.

Blueprint Gate 4 asks that the trained model stay "within 2 points" of the base
model's text scores.  Run the base model first to produce a baseline JSON, then
run the candidate with ``--candidate baseline.json``; the result records the
delta and both an absolute-within-2-points flag and a no-regression flag.

Honesty rules
-------------
* If the model, ``datasets``, a suite split, or the chat template cannot be
  loaded, the result is written with ``"status": "UNVERIFIED"`` and the exact
  reason.  It never reports an accuracy it did not measure.
* Ordering is fixed (dataset order, no shuffle), greedy generation is used,
  and the seed is recorded.  ``--offset``/``--limit`` select the same slice
  for baseline and candidate.
* Prompt scoring/formatting is documented in the functions below.  It is a
  direct subset implementation, not a claim of parity with the official
  lm-evaluation-harness configs.

Usage
-----
::

    python -m eval.text_regression --model LiquidAI/LFM3-Omni-2B \
        --suite mmlu --limit 200 --out artifacts/text_baseline.json
    python -m eval.text_regression --model checkpoints/stage3 \
        --suite mmlu --limit 200 --candidate artifacts/text_baseline.json \
        --out artifacts/text_stage3.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The processor's ChatML template, reproduced here only as a fallback when the
# tokenizer/processor does not ship a chat_template attribute.  This is the
# template in the checkpoint's processing_lfm3_omni.py, not an invention.
CHATML_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'audio' -%}{{- '<|audio|>' -}}"
    "{%- elif content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}"
    "{%- elif content['type'] == 'text' -%}{{- content['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)

MMLU_SYSTEM = (
    "You are answering a multiple-choice exam question. "
    "Reply with only the letter A, B, C, or D."
)
GSM8K_SYSTEM = (
    "Solve the math problem carefully. Show your reasoning if useful, then put "
    "the final numeric answer after a line containing only ####."
)
MMLU_CHOICES = ("A", "B", "C", "D")

GATE4_TOLERANCE_POINTS = 2.0

# A signed number, optional dollar sign, thousands separators, decimals.
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


class TextEvalUnavailable(RuntimeError):
    """The requested suite or model cannot run in this environment."""


# ---------------------------------------------------------------------------
# Pure prompt/answer helpers (unit-testable, no model)
# ---------------------------------------------------------------------------


def chat_prompt(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    """Apply the model's chat template, with the checkpoint's ChatML fallback."""
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    # The tokenizer may be a bare backend; try the processor's template once.
    try:
        from transformers import AutoProcessor  # noqa: PLC0415 - lazy

        processor = AutoProcessor.from_pretrained(
            tokenizer.name_or_path, trust_remote_code=True
        )
        if getattr(processor, "chat_template", None):
            return processor.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
    except Exception:
        # Fall through to the documented checkpoint template.
        pass
    bos = getattr(tokenizer, "bos_token", "") or ""
    rendered = [bos]
    for message in messages:
        rendered.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    rendered.append("<|im_start|>assistant\n")
    return "".join(rendered)


def mmlu_prompt(question: str, choices: Sequence[str]) -> list[dict[str, str]]:
    """MMLU prompt: question + lettered choices, answer letter requested."""
    lines = [question.strip(), ""]
    for letter, choice in zip(MMLU_CHOICES, choices):
        lines.append(f"{letter}. {str(choice).strip()}")
    lines.append("")
    lines.append("Answer with one letter:")
    return [
        {"role": "system", "content": MMLU_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def mmlu_choice_continuation(letter: str) -> str:
    """Text appended to the prompt when scoring a letter's log-likelihood."""
    return f" {letter}"


def extract_gsm8k_answer(text: str) -> str | None:
    """Return the final numeric answer as a normalised string.

    ``#### 1,234`` wins over earlier reasoning numbers.  If the model omits
    ``####`` (allowed but not prompted for), the last number in the response is
    used.  Returns ``None`` when no number is present.
    """
    if not text:
        return None
    tail = text.rsplit("####", 1)[1] if "####" in text else text
    matches = _NUMBER_RE.findall(tail)
    if not matches:
        return None
    return normalize_numeric(matches[-1])


def normalize_numeric(value: str) -> str:
    """Normalise commas, currency signs and integral decimals for exact match."""
    cleaned = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    cleaned = cleaned.replace("−", "-")
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned
    if math.isfinite(number) and number == int(number):
        return str(int(number))
    return repr(number)


def gsm8k_exact_match(prediction: str | None, gold: str | None) -> bool:
    """Exact match after numeric normalisation (GSM8K's standard check)."""
    if prediction is None or gold is None:
        return False
    return prediction == gold


def within_tolerance(
    candidate_accuracy: float | None,
    baseline_accuracy: float | None,
    *,
    tolerance_points: float = GATE4_TOLERANCE_POINTS,
) -> dict[str, Any]:
    """Compare accuracies in percentage points.

    ``passes`` is the blueprint's literal "within 2 points" (absolute delta);
    ``no_regression`` additionally states the one-sided condition.  Both are
    ``None`` when either accuracy is missing.
    """
    if candidate_accuracy is None or baseline_accuracy is None:
        return {
            "delta_points": None,
            "passes": None,
            "no_regression": None,
            "tolerance_points": tolerance_points,
        }
    delta = (float(candidate_accuracy) - float(baseline_accuracy)) * 100.0
    return {
        "delta_points": delta,
        "passes": abs(delta) <= tolerance_points + 1e-9,
        "no_regression": delta >= -tolerance_points - 1e-9,
        "tolerance_points": tolerance_points,
    }


# ---------------------------------------------------------------------------
# Model plumbing (all lazy)
# ---------------------------------------------------------------------------


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: str | None = None,
    dtype: str = "auto",
    prefer_audio: bool = False,
) -> tuple[Any, Any, Any]:
    """Load a model and tokenizer; raise a clear reason on failure.

    ``prefer_audio=True`` tries the multimodal auto classes first.  The pilot
    ASR path uses it; text regression keeps the causal-LM class first.
    """
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TextEvalUnavailable(
            "torch/transformers are not installed; install the project's training "
            "dependencies before running text regression."
        ) from error

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
    else:
        torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(dtype)
        if torch_dtype is None:
            raise TextEvalUnavailable(f"unknown dtype {dtype!r}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as error:
        raise TextEvalUnavailable(
            f"tokenizer for {model_id!r} could not be loaded: {error}"
        ) from error

    attempts: list[str] = []
    model = None
    class_order = [
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForConditionalGeneration",
        "AutoModel",
    ]
    if prefer_audio:
        class_order = [
            "AutoModelForImageTextToText",
            "AutoModelForConditionalGeneration",
            *class_order,
        ]
    for class_name in class_order:
        try:
            import transformers

            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                continue
            model = model_class.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            model.to(resolved_device)
            model.eval()
            break
        except Exception as error:  # try the next auto class
            attempts.append(f"{class_name}: {error}")
    if model is None:
        raise TextEvalUnavailable(
            f"could not load a text-capable model for {model_id!r}: "
            + " | ".join(attempts[-2:])
        )
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, torch.device(resolved_device)


def _model_logits(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, (tuple, list)):
        logits = outputs[0]
    if logits is None:
        raise TextEvalUnavailable(
            "model forward returned no logits; use a checkpoint with a text LM head"
        )
    return logits


def _tokenize(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def score_choice_logprobs(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    prompt: str,
    continuations: Sequence[str],
) -> list[float]:
    """Summed log-probability of each continuation after ``prompt``."""
    prompt_ids = _tokenize(tokenizer, prompt, add_special_tokens=False)
    if not prompt_ids:
        raise TextEvalUnavailable("empty prompt after tokenisation")
    scores: list[float] = []
    with torch.no_grad():
        for continuation in continuations:
            continuation_ids = _tokenize(
                tokenizer, continuation, add_special_tokens=False
            )
            if not continuation_ids:
                scores.append(float("-inf"))
                continue
            input_ids = torch.tensor(
                [prompt_ids + continuation_ids], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids)
            logits = _model_logits(model, input_ids, attention_mask)
            start = len(prompt_ids) - 1
            selected = logits[:, start : start + len(continuation_ids), :]
            log_probs = torch.log_softmax(selected.float(), dim=-1)
            target = torch.tensor(
                [continuation_ids], dtype=torch.long, device=device
            ).unsqueeze(-1)
            score = log_probs.gather(-1, target).sum().item()
            scores.append(float(score))
    return scores


def generate_text(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    """Greedy generation for GSM8K."""
    prompt_ids = _tokenize(tokenizer, prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    sequences = getattr(output, "sequences", output)
    if hasattr(sequences, "tolist"):
        sequences = sequences.tolist()
    generated = sequences[0][len(prompt_ids) :]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def load_suite(suite: str, *, offset: int, limit: int | None) -> list[dict[str, Any]]:
    """Load a deterministic slice of MMLU (test) or GSM8K (test)."""
    try:
        from datasets import load_dataset  # noqa: PLC0415 - lazy
    except ImportError as error:
        raise TextEvalUnavailable(
            "the `datasets` package is not installed; install project dependencies "
            "or pip install datasets."
        ) from error
    try:
        if suite == "mmlu":
            dataset = load_dataset("cais/mmlu", "all", split="test")
        elif suite == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split="test")
        else:
            raise TextEvalUnavailable(f"unknown suite {suite!r}")
    except TextEvalUnavailable:
        raise
    except Exception as error:
        raise TextEvalUnavailable(
            f"could not load the {suite} dataset: {error}. This may be a network, "
            "authentication, or dataset-name problem; no score was produced."
        ) from error

    rows: list[dict[str, Any]] = []
    end = None if limit is None else offset + limit
    for index in range(offset, min(len(dataset), end if end is not None else len(dataset))):
        row = dict(dataset[index])
        row["_index"] = index
        rows.append(row)
    if not rows:
        raise TextEvalUnavailable(
            f"{suite}: offset {offset} and limit {limit} selected no examples"
        )
    return rows


def run_mmlu(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    rows: Sequence[dict[str, Any]],
    *,
    limit: int | None,
) -> dict[str, Any]:
    letters = list(MMLU_CHOICES)
    per_example: list[dict[str, Any]] = []
    correct = 0
    for position, row in enumerate(rows):
        if limit is not None and position >= limit:
            break
        question = str(row.get("question", ""))
        choices = list(row.get("choices", []))
        if len(choices) != 4:
            continue
        gold_index = int(row["answer"])
        prompt = chat_prompt(tokenizer, mmlu_prompt(question, choices))
        scores = score_choice_logprobs(
            model,
            tokenizer,
            torch,
            device,
            prompt,
            [mmlu_choice_continuation(letter) for letter in letters],
        )
        predicted_index = max(range(4), key=lambda index: scores[index])
        is_correct = predicted_index == gold_index
        correct += int(is_correct)
        per_example.append(
            {
                "index": row["_index"],
                "gold": letters[gold_index],
                "predicted": letters[predicted_index],
                "correct": is_correct,
                "logprobs": [round(value, 6) for value in scores],
            }
        )
    total = len(per_example)
    return {
        "accuracy": (correct / total) if total else None,
        "correct": correct,
        "examples": total,
        "per_example": per_example,
    }


def run_gsm8k(
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    rows: Sequence[dict[str, Any]],
    *,
    limit: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    per_example: list[dict[str, Any]] = []
    correct = 0
    for position, row in enumerate(rows):
        if limit is not None and position >= limit:
            break
        question = str(row.get("question", ""))
        gold = extract_gsm8k_answer(str(row.get("answer", "")))
        prompt = chat_prompt(
            tokenizer,
            [
                {"role": "system", "content": GSM8K_SYSTEM},
                {"role": "user", "content": question},
            ],
        )
        completion = generate_text(
            model,
            tokenizer,
            torch,
            device,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        prediction = extract_gsm8k_answer(completion)
        is_correct = gsm8k_exact_match(prediction, gold)
        correct += int(is_correct)
        per_example.append(
            {
                "index": row["_index"],
                "gold": gold,
                "predicted": prediction,
                "correct": is_correct,
                "completion": completion,
            }
        )
    total = len(per_example)
    return {
        "accuracy": (correct / total) if total else None,
        "correct": correct,
        "examples": total,
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def gate4_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Gate-4 artifact expected by ``src.train_stage2``.

    ``src.train_stage2.check_gate4`` accepts ``gate`` + ``result`` (or
    ``passed``).  A missing/unverified comparison is written as UNVERIFIED and
    ``passed: false`` so stage 2 refuses it instead of guessing.
    """
    comparison = result.get("comparison")
    artifact: dict[str, Any] = {
        "gate": "Gate 4",
        "metric": "text_regression",
        "suite": result.get("suite"),
        "model": result.get("model"),
        "examples": result.get("examples"),
        "offset": result.get("offset"),
        "limit": result.get("limit"),
        "seed": result.get("seed"),
        "accuracy": result.get("accuracy"),
    }
    if not isinstance(comparison, Mapping):
        artifact.update(
            {
                "result": "UNVERIFIED",
                "passed": False,
                "reason": "no --candidate baseline results supplied",
                "delta_points": None,
            }
        )
        return artifact
    artifact.update(
        {
            "baseline": comparison.get("baseline"),
            "baseline_accuracy": comparison.get("baseline_accuracy"),
            "candidate_accuracy": comparison.get("candidate_accuracy"),
            "delta_points": comparison.get("delta_points"),
            "tolerance_points": comparison.get(
                "tolerance_points", GATE4_TOLERANCE_POINTS
            ),
            "passes": comparison.get("passes"),
            "no_regression": comparison.get("no_regression"),
        }
    )
    if comparison.get("status") == "ok" and comparison.get("passes") is not None:
        passed = bool(comparison.get("passes"))
        artifact["result"] = "PASS" if passed else "FAIL"
        artifact["passed"] = passed
    else:
        artifact["result"] = "UNVERIFIED"
        artifact["passed"] = False
        artifact["reason"] = comparison.get("reason", "comparison unverified")
    return artifact


def run(
    *,
    model_id: str,
    suite: str,
    out: Path,
    limit: int | None,
    offset: int,
    candidate: Path | None,
    seed: int,
    device: str | None,
    dtype: str,
    max_new_tokens: int,
    gate_artifact: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "UNVERIFIED",
        "suite": suite,
        "model": model_id,
        "limit": limit,
        "offset": offset,
        "seed": seed,
        "examples": 0,
        "accuracy": None,
    }
    _set_deterministic(seed)
    try:
        rows = load_suite(suite, offset=offset, limit=limit)
        model, tokenizer, torch_device = load_model_and_tokenizer(
            model_id, device=device, dtype=dtype
        )
        torch_module = __import__("torch")
        if suite == "mmlu":
            metrics = run_mmlu(
                model, tokenizer, torch_module, torch_device, rows, limit=limit
            )
        elif suite == "gsm8k":
            metrics = run_gsm8k(
                model,
                tokenizer,
                torch_module,
                torch_device,
                rows,
                limit=limit,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise TextEvalUnavailable(f"unknown suite {suite!r}")
        result.update(metrics)
        result["status"] = "ok"
    except TextEvalUnavailable as error:
        result["reason"] = str(error)
    except Exception as error:  # keep an honest record instead of crashing mid-suite
        result["reason"] = f"{type(error).__name__}: {error}"

    if candidate is not None:
        result["comparison"] = compare_candidate(result, candidate)
    if candidate is not None or gate_artifact is not None:
        gate = gate4_artifact(result)
        result.update({"gate": gate["gate"], "result": gate["result"], "passed": gate["passed"]})
        if gate_artifact is not None:
            gate_artifact.parent.mkdir(parents=True, exist_ok=True)
            gate_artifact.write_text(
                json.dumps(gate, indent=2) + "\n", encoding="utf-8"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare_candidate(
    candidate_result: dict[str, Any], baseline_path: Path
) -> dict[str, Any]:
    """Compare the just-run candidate against a prior baseline result JSON."""
    comparison: dict[str, Any] = {
        "baseline": str(baseline_path),
        "status": "UNVERIFIED",
        "delta_points": None,
        "passes": None,
        "no_regression": None,
        "tolerance_points": GATE4_TOLERANCE_POINTS,
    }
    if not baseline_path.exists():
        comparison["reason"] = f"baseline result not found: {baseline_path}"
        return comparison
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        comparison["reason"] = f"baseline is not valid JSON: {error}"
        return comparison
    if baseline.get("status") != "ok" or candidate_result.get("status") != "ok":
        comparison["reason"] = (
            "baseline or candidate is UNVERIFIED; refusing to compute a delta"
        )
        return comparison
    if baseline.get("suite") != candidate_result.get("suite"):
        comparison["reason"] = (
            f"suite mismatch: baseline={baseline.get('suite')!r} "
            f"candidate={candidate_result.get('suite')!r}"
        )
        return comparison
    if (baseline.get("offset"), baseline.get("limit")) != (
        candidate_result.get("offset"),
        candidate_result.get("limit"),
    ):
        comparison["reason"] = (
            "offset/limit mismatch; the two runs did not evaluate the same subset"
        )
        return comparison
    tolerance = within_tolerance(
        candidate_result.get("accuracy"), baseline.get("accuracy")
    )
    comparison.update(tolerance)
    comparison["baseline_accuracy"] = baseline.get("accuracy")
    comparison["candidate_accuracy"] = candidate_result.get("accuracy")
    comparison["status"] = "ok"
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.text_regression",
        description=(
            "Direct deterministic MMLU (A-D log-likelihood) or GSM8K (greedy "
            "generation + final-answer exact match) subset evaluation."
        ),
    )
    parser.add_argument("--model", required=True, help="Base model id or checkpoint path.")
    parser.add_argument("--suite", choices=("mmlu", "gsm8k"), required=True)
    parser.add_argument("--limit", type=int, default=200, help="Number of examples.")
    parser.add_argument("--offset", type=int, default=0, help="Start index (fixed order).")
    parser.add_argument("--out", type=Path, default=Path("artifacts/text_results.json"))
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Prior baseline results JSON to compare this run against.",
    )
    parser.add_argument(
        "--gate-artifact",
        type=Path,
        default=None,
        help=(
            "Write a Gate-4 artifact ({gate, result: PASS/FAIL/UNVERIFIED, "
            "passed, delta_points, ...}) for src.train_stage2 --gate4."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, cuda:1, cpu, ...")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    result = run(
        model_id=args.model,
        suite=args.suite,
        out=args.out,
        limit=args.limit,
        offset=args.offset,
        candidate=args.candidate,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        gate_artifact=args.gate_artifact,
    )
    if result["status"] != "ok":
        print(f"UNVERIFIED: {result.get('reason')}", file=sys.stderr)
        return 2
    print(
        f"{args.suite}: {result['correct']}/{result['examples']} "
        f"= {result['accuracy']:.4f}; wrote {args.out}"
    )
    comparison = result.get("comparison")
    if comparison and comparison.get("status") == "ok":
        print(
            f"delta {comparison['delta_points']:+.2f} points; "
            f"within 2 points: {comparison['passes']}; "
            f"no regression: {comparison['no_regression']}"
        )
    elif comparison:
        print(f"comparison UNVERIFIED: {comparison.get('reason')}", file=sys.stderr)
    if comparison and comparison.get("status") == "ok" and comparison.get("passes") is False:
        return 1
    if args.gate_artifact is not None:
        print(f"wrote Gate-4 artifact {args.gate_artifact} ({result.get('result')})")
        if result.get("passed") is not True:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
