"""
Train script:
- reads data/raw/Phishing_URL_Dataset.csv (or processed CSV if present)
- auto-detects URL column & label column
- builds features, scales, trains SimpleTabNet
- saves artifacts to src/serve/artifacts/: model.pt, scaler.pkl, feature_keys.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from .tabm_adapters import RefTABMNet
from ...features.reference_features import build_reference_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_CSV = os.path.join(ROOT, "data", "raw", "Phishing_URL_Dataset.csv")
PROC_CSV = os.path.join(ROOT, "data", "processed", "processed_phishing_urls.csv")
ART_DIR = os.path.join(ROOT, "src", "serve", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

def _pick_cols(df: pd.DataFrame) -> Tuple[str, str]:
    # guess url column
    url_candidates = [c for c in df.columns if c.lower() in {"url", "urls", "link", "domain"}]
    if not url_candidates:
        raise ValueError("Could not find URL column (expected one of: url, urls, link, domain).")
    url_col = url_candidates[0]

    # guess label column
    label_candidates = [c for c in df.columns if c.lower() in {"label", "class", "target", "phishing"}]
    if not label_candidates:
        raise ValueError("Could not find label column (expected one of: label, class, target, phishing).")
    y_col = label_candidates[0]
    return url_col, y_col

def featurize(urls: List[str]) -> Tuple[np.ndarray, List[str]]:
    rows = [build_reference_features(u) for u in urls]
    # fixed order of features
    keys = [
        "https","has_ip","len_url","len_host","len_path","num_dots_host","num_hyphen_host",
        "num_digits_url","num_query_params","has_at","has_frag","path_depth",
        "ref_phish_hits","ref_benign_hits","ref_ratio"
    ]
    X = np.array([[float(r.get(k, 0.0)) for k in keys] for r in rows], dtype=np.float32)
    return X, keys

def train():
    csv_path = PROC_CSV if os.path.exists(PROC_CSV) else RAW_CSV
    df = pd.read_csv(csv_path)
    url_col, y_col = _pick_cols(df)

    # normalize labels to {0,1}
    y = df[y_col]
    if y.dtype == object:
        y = y.str.lower().map({"phish":1,"phishing":1,"malicious":1,"bad":1,"benign":0,"legit":0,"legitimate":0})
    y = y.astype(int).values

    X_raw, feature_keys = featurize(df[url_col].astype(str).tolist())

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # model
    input_dim = X.shape[1]
    model = RefTABMNet(input_dim=input_dim)
    torch_model = model.model

    opt = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    epochs = 30
    torch_model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = torch_model(X_tr_t).squeeze(-1)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()

    # eval
    torch_model.eval()
    with torch.inference_mode():
        logits_va = torch_model(X_va_t).squeeze(-1)
        probs_va = torch.sigmoid(logits_va).cpu().numpy()
    auc = roc_auc_score(y_va, probs_va)
    pred = (probs_va >= 0.5).astype(int)
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(y_va, pred, average="binary")

    print(f"AUC={auc:.4f}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # save artifacts
    torch.save(torch_model.state_dict(), os.path.join(ART_DIR, "model.pt"))
    joblib.dump(scaler, os.path.join(ART_DIR, "scaler.pkl"))
    with open(os.path.join(ART_DIR, "feature_keys.json"), "w") as f:
        json.dump(feature_keys, f)

if __name__ == "__main__":
    train()
