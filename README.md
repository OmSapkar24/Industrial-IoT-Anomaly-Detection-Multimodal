# Multimodal Anomaly Detection for Industrial IoT Sensors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![TorchAudio](https://img.shields.io/badge/TorchAudio-0.12+-orange)](https://pytorch.org/audio/stable/index.html)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-informational)](https://www.docker.com/)

End-to-end anomaly detection combining time-series telemetry, vibration, and acoustic data using temporal CNNs, transformers, and spectrogram encoders.

## Overview
- Sensor fusion: time-series (TFT/TCN), vibration (1D CNN), audio (Mel-spectrogram + CNN)
- Self-supervised pretraining (TS2Vec/SimCLR variants) to leverage unlabeled data
- Online detection with adaptive thresholds (POT/EVA) and concept drift monitors
- Edge-ready lightweight model for on-device detection

## Business Context
- Reduce unplanned downtime and maintenance costs (PdM)
- Early detection of bearing failures, cavitation, misalignment, leaks
- KPIs: >95% recall for critical faults, <2% false alarm rate

## Tech Stack
- Data: PyTorch, TorchAudio, Pandas, Scipy
- Modeling: TFT/Transformer, Temporal CNN, EfficientNet-like CNN for spectrograms
- Serving: FastAPI, ONNX export, Docker
- MLOps: MLflow, DVC, GitHub Actions

## Repository Structure
```
iiot-anomaly/
  data/
  notebooks/
  src/
    data/
      load_timeseries.py
      audio_utils.py
    models/
      tcn.py
      tft.py
      audio_encoder.py
      fusion.py
    train.py
    infer.py
  serving/
    api.py
  configs/
    model.yaml
  tests/
  docker/
    Dockerfile
```

## Sample Results (synthetic)
- AUROC: 0.972 (fusion) vs 0.914 (time-series only)
- F1@threshold (critical faults): 0.91
- Latency P95: 22ms per window on CPU

## Installation
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Train:
```
python -m src.train --config configs/model.yaml
```
Online inference API:
```
uvicorn serving.api:app --host 0.0.0.0 --port 8080
```

## Roadmap
- Distributed training for large fleets
- Active learning loop with operator feedback
- TinyML variant (TFLite/Edge Impulse)
