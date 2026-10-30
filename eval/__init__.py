"""Offline evaluation harnesses for Repartee.

Modules
-------
``eval.metrics``
    Pure, interval-based Full-Duplex-Bench (FDB) turn-taking metrics.
``eval.fdb``
    FDB v1.0 / v1.5 annotation runner (CLI: ``python -m eval.fdb``).
``eval.text_regression``
    Direct MMLU / GSM8K subset evaluation (CLI: ``python -m eval.text_regression``).
``eval.pilot``
    20-scenario live pilot (CLI: ``python -m eval.pilot``).

Heavy dependencies (torch, transformers, datasets, silero-vad) are imported
lazily inside the code paths that need them, so importing this package is cheap
and works on a CPU-only host.
"""

from __future__ import annotations

__all__ = ["metrics"]
