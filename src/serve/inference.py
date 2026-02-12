import os
import json
import joblib
import numpy as np
from typing import Dict, Tuple
from .models.tabm_adapters import RefTABMNet
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
    if _keys is None:
        with open(KEYS_JSON, "r") as f:
            _keys = json.load(f)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PKL)
    if _model is None:
        _model = RefTABMNet(input_dim=len(_keys), model_path=MODEL_PT, device="cpu")

def _featurize(url: str) -> Tuple[np.ndarray, Dict]:
    ref = build_reference_features(url)
    vec = np.array([[float(ref.get(k, 0.0)) for k in _keys]], dtype=np.float32)
    vec = _scaler.transform(vec)
    return vec, ref

def predict(url: str) -> Dict:
    _ensure_loaded()
    X, debug = _featurize(url)
    proba = float(_model.predict_proba(X)[0])
    label = "phish" if proba >= 0.5 else "benign"
    return {"label": label, "score": proba, "details": debug}
