# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install .
ENTRYPOINT ["repartee"]

# USB receiver: stdlib only on Linux; no model, no GPU.
FROM base AS receiver

# Bigger device: LFM3-Omni-2B, torch, soundfile, librosa, PyYAML.
FROM base AS worker
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" '.[bigger]'
