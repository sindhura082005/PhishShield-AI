import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from ...features.reference_features import build_reference_features

print("RUNNING NEW TRAINING FILE")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_CSV = os.path.join(ROOT, "data", "raw", "Phishing_URL_Dataset.csv")
PROC_CSV = os.path.join(ROOT, "data", "processed", "processed_phishing_urls.csv")
ART_DIR = os.path.join(ROOT, "src", "serve", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


def _pick_cols(df: pd.DataFrame) -> Tuple[str, str]:
    url_candidates = [c for c in df.columns if c.lower() in {"url", "urls", "link", "domain"}]
    label_candidates = [c for c in df.columns if c.lower() in {"label", "class", "target", "phishing"}]

    if not url_candidates:
        raise ValueError("URL column not found")
    if not label_candidates:
        raise ValueError("Label column not found")

    return url_candidates[0], label_candidates[0]


def featurize(urls: List[str]) -> Tuple[np.ndarray, List[str]]:
    rows = [build_reference_features(u) for u in urls]

    keys = [
        "https","has_ip","len_url","len_host","len_path","num_dots_host",
        "num_hyphen_host","num_digits_url","num_query_params","has_at",
        "has_frag","path_depth","ref_phish_hits","ref_benign_hits","ref_ratio"
    ]

    X = np.array([[float(r.get(k, 0.0)) for k in keys] for r in rows], dtype=np.float32)
    return X, keys


def train():

    csv_path = PROC_CSV if os.path.exists(PROC_CSV) else RAW_CSV
    df = pd.read_csv(csv_path)

    print("Dataset size:", len(df))

    url_col, y_col = _pick_cols(df)

    y = df[y_col]
    if y.dtype == object:
        y = y.str.lower().map({
            "phish":1,"phishing":1,"malicious":1,"bad":1,
            "benign":0,"legit":0,"legitimate":0
        })

    y = y.astype(int).values

    X_raw, feature_keys = featurize(df[url_col].astype(str).tolist())

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    input_dim = X.shape[1]

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

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_tr),
        y=y_tr
    )

    pos_weight = torch.tensor(class_weights[1] / class_weights[0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_auc = 0
    patience_counter = 0
    epochs = 100

    for ep in range(epochs):

        model.train()
        optimizer.zero_grad()

        logits = model(X_tr_t).squeeze(-1)
        loss = loss_fn(logits, y_tr_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).numpy()

        auc = roc_auc_score(y_va, val_probs)
        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % 10 == 0:
            print(f"Epoch {ep} | Loss {loss.item():.4f} | Val AUC {auc:.4f}")

        if patience_counter > 10:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        logits_va = model(X_va_t).squeeze(-1)
        probs_va = torch.sigmoid(logits_va).numpy()

    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.3, 0.7, 0.01):
        pred = (probs_va >= thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_va, pred, average="binary"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    final_pred = (probs_va >= best_thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_va, final_pred, average="binary"
    )
    auc = roc_auc_score(y_va, probs_va)

    print("\nFINAL RESULTS")
    print(f"AUC = {auc:.4f}")
    print(f"Precision = {p:.4f}")
    print(f"Recall = {r:.4f}")
    print(f"F1 = {f1:.4f}")
    print(f"Best Threshold = {best_thresh:.2f}")

    torch.save(model.state_dict(), os.path.join(ART_DIR, "model.pt"))
    joblib.dump(scaler, os.path.join(ART_DIR, "scaler.pkl"))

    with open(os.path.join(ART_DIR, "feature_keys.json"), "w") as f:
        json.dump(feature_keys, f)


if __name__ == "__main__":
    train()
