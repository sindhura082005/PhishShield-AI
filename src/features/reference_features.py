import re
from collections import Counter
from typing import Dict
from .preprocess import extract_basic_url_features

# lightweight token dictionaries; you can expand from data stats during training
BENIGN_HINTS = {"about", "contact", "home", "help", "docs", "support", "community", "blog", "shop"}
PHISH_HINTS  = {"verify", "confirm", "update", "secure", "account", "wallet", "password", "bank", "login"}

def build_reference_features(url: str) -> Dict[str, float]:
    base = extract_basic_url_features(url)
    toks = [t for t in re.split(r"[\W_]+", url.lower()) if t]
    counts = Counter(toks)
    base["ref_phish_hits"] = float(sum(counts[t] for t in PHISH_HINTS if t in counts))
    base["ref_benign_hits"] = float(sum(counts[t] for t in BENIGN_HINTS if t in counts))
    # heuristic: ratio (avoid div-by-zero)
    denom = base["ref_benign_hits"] + 1.0
    base["ref_ratio"] = base["ref_phish_hits"] / denom
    return base
