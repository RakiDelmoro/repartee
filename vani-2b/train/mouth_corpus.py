#!/usr/bin/env python3
"""Build the mouth fine-tune clause corpus (spoken forms, train/holdout split)."""

import itertools
import json
import random
from pathlib import Path

OUT = Path("data/probe/mouth")
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)

names = [
    "Doctor Okafor", "Priya Raman", "Raki Delmoro", "Sam Whitfield",
    "Tom Alvarez", "Maya Chen", "Elena Rossi", "Jamal Carter",
    "Aisha Khan", "Wei Zhang",
]
projects = [
    "the Harrow project", "the Q2 launch", "the enterprise tier",
    "the vendor review", "the staging plan", "the audit logs",
    "the billing migration", "the mobile release", "the data warehouse",
    "the support tooling",
]
amounts = [
    "three point two million pounds", "forty thousand dollars",
    "two hundred and fifty thousand euros", "one point eight million dollars",
    "nine hundred thousand dollars", "twelve million pounds",
]
dates = [
    "the twenty second of November", "Friday", "the second week of April",
    "the end of the quarter", "Monday morning", "the first of March",
    "next Tuesday", "the end of next month",
]
documents = [
    "revised estimate", "vendor termination clause", "launch plan",
    "pricing sheet", "security review", "risk register",
    "migration checklist", "support runbook",
]
quarters = ["Q2", "the second quarter", "next quarter"]

templates = {
    "decision": [
        "{project} moves to {quarter}.",
        "{owner} approved {amount} for {project}.",
        "We are delaying {project} until {date}.",
        "The team decided to pause {project} until {date}.",
        "The board approved {amount} for {project}.",
        "{project} is on hold until {date}.",
        "We are keeping {project} in {quarter}.",
    ],
    "action": [
        "{owner} will send the {document} by {date}.",
        "Get legal to review the {document} by {date}.",
        "{owner} owns the {document}.",
        "Please ask {owner} to update the {document} before {date}.",
        "{owner} will review the {document}.",
        "Ask {owner} for the {document} by {date}.",
        "{owner} should call the vendor about {project}.",
    ],
    "question": [
        "Do we still need the {document}?",
        "Who is covering {project}?",
        "Did {owner} sign off on the {document}?",
        "Has {owner} seen the {document}?",
        "Are we still shipping {project} in {quarter}?",
    ],
    "topic": [
        "Pricing for {project} is parked.",
        "The {document} came up again.",
        "We spent most of the meeting on {project}.",
        "We talked about {project} and the {document}.",
        "The {document} is the main topic.",
    ],
    "filler": [
        "Okay, perfect.",
        "That makes sense.",
        "Let me check that.",
        "Right, got it.",
        "Sure, one moment.",
        "Thanks, that helps.",
    ],
    "numeric": [
        "The estimate is {amount}.",
        "Revenue was {amount} last quarter.",
        "The budget is {amount}.",
        "The contract is worth {amount}.",
        "We spent {amount} on {project}.",
    ],
    "date": [
        "The deadline is {date}.",
        "The review is on {date}.",
        "We need it by {date}.",
        "We meet again on {date}.",
        "The deadline moved to {date}.",
    ],
}

HOLDOUT_FAMILIES = {"question", "topic", "date"}
HOLDOUT_NAME = "Maya Chen"

rows = []
for family, family_templates in templates.items():
    for template in family_templates:
        for _ in range(40):
            text = template.format(
                owner=rng.choice(names), project=rng.choice(projects),
                amount=rng.choice(amounts), date=rng.choice(dates),
                document=rng.choice(documents), quarter=rng.choice(quarters),
            )
            if text in {r["text"] for r in rows}:
                continue
            holdout = family in HOLDOUT_FAMILIES or HOLDOUT_NAME in text
            rows.append({"text": text, "family": family, "split": "holdout" if holdout else "train"})
            if len(rows) >= 700:
                break
        if len(rows) >= 700:
            break
    if len(rows) >= 700:
        break

# Add the real clauses from the knowledge base, spoken form left as written.
import sqlite3
db = sqlite3.connect("data/repartee.db")
for (text,) in db.execute("select text from events"):
    if text and text not in {r["text"] for r in rows}:
        rows.append({"text": text, "family": "kb", "split": "holdout" if len(rows) % 5 == 0 else "train"})

rng.shuffle(rows)
for i, row in enumerate(rows):
    row["id"] = f"m{i:04d}"

with (OUT / "corpus.jsonl").open("w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

train = sum(r["split"] == "train" for r in rows)
print(f"clauses={len(rows)} train={train} holdout={len(rows)-train}")
print("families:", {f: sum(r["family"] == f for r in rows) for f in sorted({r['family'] for r in rows})})
for r in rows[:5]:
    print(r["split"], r["family"], r["text"])
