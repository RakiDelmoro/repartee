"""Knowledge-base extraction: transcript -> structured events.

M2 of docs/knowledge-base.md. Talks to any OpenAI-compatible chat endpoint
(llama.cpp server, ollama, vLLM) and never trusts the model's output unchecked:
every event must prove itself with a quote that exists in the transcript.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from . import config

log = logging.getLogger(__name__)

KINDS = ("decision", "action", "question", "topic")

SYSTEM_PROMPT = """You extract structure from meeting transcripts.

The input is raw automatic speech recognition output: one continuous paragraph,
no speaker labels, no timestamps, punctuation and filler words included.

Rules:
1. Return JSON only, with exactly these keys: decisions, actions, questions, topics.
2. Every decision, action and question MUST include "evidence": a sentence copied
   verbatim from the transcript. If you cannot copy an exact sentence, omit the item.
3. "owner" is a person's name only when that name appears inside the evidence you
   quoted. If the name is not in the quoted sentence, use null. Never infer an
   owner from "I will" or "do it".
4. Never invent a due date. Use null unless one was said.
5. "questions" means explicit interrogative sentences only ("Do we...", "Should
   we...", "What about..."). Do not turn statements into questions.
6. Ignore small talk, logistics and status updates that contain no outcome. A
   statement of fact ("we are four weeks behind") is not a decision.
7. At most 5 decisions, 5 actions, 3 questions. Keep "text" under 20 words and
   "evidence" under 30 words.
8. If the transcript contains no decisions or actions, return empty arrays. That is
   a correct answer, not a failure.

Output shape:
{
  "decisions": [{"text": "...", "evidence": "..."}],
  "actions":   [{"text": "...", "owner": null, "due": null, "evidence": "..."}],
  "questions": [{"text": "...", "evidence": "..."}],
  "topics":    ["..."]
}"""


@dataclass(frozen=True)
class ExtractedEvent:
    kind: str
    text: str
    evidence: str
    owner: str | None = None
    due: str | None = None
    t_start: float | None = None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _locate(evidence: str, source: str) -> int | None:
    """Character offset of the evidence in the source, or None."""
    words = _normalise(evidence).split()[:6]
    if not words:
        return None
    pattern = r"\W+".join(re.escape(word) for word in words)
    match = re.search(pattern, source, re.IGNORECASE)
    return match.start() if match else None


def owner_is_supported(owner: str | None, evidence: str) -> bool:
    """An owner is only trusted when the quoted sentence names them."""
    if not owner:
        return True
    return _normalise(owner) in _normalise(evidence)


def evidence_is_grounded(evidence: str, source: str) -> bool:
    """True when the quote can be found in the transcript.

    Exact match first; then a tolerance for ASR punctuation and one-word drift,
    by requiring the quote's words to appear in order.
    """
    quote, text = _normalise(evidence), _normalise(source)
    if not quote:
        return False
    if quote in text:
        return True

    needle = quote.split()
    haystack = text.split()
    if not needle or not haystack:
        return False

    matched = 0
    cursor = 0
    for word in needle:
        while cursor < len(haystack) and haystack[cursor] != word:
            cursor += 1
        if cursor == len(haystack):
            break
        matched += 1
        cursor += 1
    return matched / len(needle) >= 0.9


def chunk_transcript(text: str, max_chars: int = 6000) -> list[str]:
    """Split on sentence boundaries, never mid-sentence."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def build_messages(chunk: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcript:\n\n{chunk}"},
    ]


class TruncatedResponse(RuntimeError):
    """The model ran out of tokens before finishing."""


class LLMClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        api_key: str = config.LLM_API_KEY,
        timeout: float = config.LLM_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, messages: Iterable[dict[str, str]], max_tokens: int = 2048) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                # Qwen3.5 reasons by default; that burns the budget before any JSON.
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())

        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            raise TruncatedResponse(choice["message"].get("content", "")[:200])
        return choice["message"]["content"]


def parse_response(raw: str) -> dict[str, Any]:
    """Tolerate a markdown fence or leading prose, but require an object."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    return json.loads(text[start : end + 1])


def to_events(parsed: dict[str, Any], source: str) -> list[ExtractedEvent]:
    """Convert a parsed response into events, dropping anything not grounded."""
    events: list[ExtractedEvent] = []

    for key, kind in (("decisions", "decision"), ("questions", "question")):
        for item in parsed.get(key) or []:
            text = (item.get("text") or "").strip()
            evidence = (item.get("evidence") or "").strip()
            if text and evidence_is_grounded(evidence, source):
                events.append(ExtractedEvent(kind=kind, text=text, evidence=evidence))
            elif text:
                log.warning("dropped %s with ungrounded evidence: %s", kind, evidence[:80])

    for item in parsed.get("actions") or []:
        text = (item.get("text") or "").strip()
        evidence = (item.get("evidence") or "").strip()
        if text and evidence_is_grounded(evidence, source):
            owner = (item.get("owner") or None)
            if not owner_is_supported(owner, evidence):
                log.info("owner %r is not named in the quote; setting null", owner)
                owner = None
            events.append(
                ExtractedEvent(
                    kind="action",
                    text=text,
                    evidence=evidence,
                    owner=owner,
                    due=(item.get("due") or None),
                )
            )
        elif text:
            log.warning("dropped action with ungrounded evidence: %s", evidence[:80])

    for topic in parsed.get("topics") or []:
        topic = (topic or "").strip()
        if topic:
            events.append(ExtractedEvent(kind="topic", text=topic, evidence=topic))

    return events


def _merge_overlapping(events: list[ExtractedEvent], source: str) -> list[ExtractedEvent]:
    """Two events quoting the same passage are one event.

    Decisions also merge when their quotes are adjacent: models tend to split a
    compound decision ("we won't renew ... we'll start with staging") into two.
    Actions stay separate even when adjacent - they are distinct tasks.
    """
    spans: list[int | None] = []
    out: list[ExtractedEvent] = []

    for event in events:
        span = _locate(event.evidence, source)
        merged = False
        if span is not None:
            end = span + len(_normalise(event.evidence))
            for index, kept in enumerate(out):
                kept_span = spans[index]
                if kept_span is None or kept.kind != event.kind:
                    continue
                kept_end = kept_span + len(_normalise(kept.evidence))
                overlaps = span < kept_end and kept_span < end
                adjacent_decision = (
                    kept.kind == "decision"
                    and span - kept_end <= 40
                    and kept_span - end <= 40
                )
                if overlaps or adjacent_decision:
                    out[index] = ExtractedEvent(
                        kind=kept.kind,
                        text=(
                            event.text
                            if event.text in kept.text
                            else f"{kept.text.rstrip(' .;')}; {event.text}"
                        ),
                        evidence=max(kept.evidence, event.evidence, key=len),
                        owner=kept.owner or event.owner,
                        due=kept.due or event.due,
                    )
                    merged = True
                    break
        if not merged:
            out.append(event)
            spans.append(span)
    return out


def extract(transcript: str, client: LLMClient | None = None) -> list[ExtractedEvent]:
    """Chunk, extract, and merge into one event list."""
    client = client or LLMClient()
    events: list[ExtractedEvent] = []

    for chunk in chunk_transcript(transcript):
        raw = None
        for budget in (2048, 8192):
            try:
                raw = client.complete(build_messages(chunk), max_tokens=budget)
                break
            except TruncatedResponse:
                log.warning("response truncated at %d tokens; retrying", budget)
        if raw is None:
            log.error("skipping chunk: model kept truncating")
            continue

        try:
            parsed = parse_response(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("unparseable response, skipping chunk: %s", exc)
            continue

        events.extend(_merge_overlapping(to_events(parsed, chunk), chunk))

    return _dedupe(events)


def _dedupe(events: list[ExtractedEvent]) -> list[ExtractedEvent]:
    seen: set[tuple[str, str]] = set()
    out: list[ExtractedEvent] = []
    for event in events:
        key = (event.kind, _normalise(event.text))
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out
