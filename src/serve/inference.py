import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Tuple
from ..features.reference_features import build_reference_features

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, "artifacts")

MODEL_PT = os.path.join(ART, "model.pt")
SCALER_PKL = os.path.join(ART, "scaler.pkl")
KEYS_JSON = os.path.join(ART, "feature_keys.json")

_model = None
_scaler = None
_keys = None


def _ensure_loaded():
    global _model, _scaler, _keys

    # Load feature keys
    if _keys is None:
        with open(KEYS_JSON, "r") as f:
            _keys = json.load(f)

    # Load scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PKL)

    # Load model (SAME architecture as training)
    if _model is None:
        input_dim = len(_keys)

        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

        model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
        model.eval()

        _model = model


def _featurize(url: str) -> Tuple[np.ndarray, Dict]:
    ref = build_reference_features(url)

    vec = np.array(
        [[float(ref.get(k, 0.0)) for k in _keys]],
        dtype=np.float32
    )

    vec = _scaler.transform(vec)

    return vec, ref


def predict(url: str) -> Dict:
    _ensure_loaded()

    X, debug = _featurize(url)

    with torch.no_grad():
        logits = _model(torch.tensor(X, dtype=torch.float32))
        proba = torch.sigmoid(logits).item()

    label = "phish" if proba >= 0.5 else "benign"

    return {
        "label": label,
        "score": float(proba),
        "details": debug
    }
