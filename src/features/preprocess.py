import re
import ipaddress
from urllib.parse import urlparse
from typing import Dict

def _is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host)
        return 1
    except Exception:
        return 0

def extract_basic_url_features(url: str) -> Dict[str, float]:
    """
    General, model-agnostic URL features. Keep this PURE (no global state).
    """
    try:
        p = urlparse(url.strip())
    except Exception:
        # fall back: treat entire string as path
        p = urlparse("http://" + url.strip())

    scheme = (p.scheme or "").lower()
    host = (p.netloc or "").lower()
    path = (p.path or "").lower()
    query = p.query or ""
    fragment = p.fragment or ""

    host_no_port = host.split(":")[0]
    features = {
        "https": 1.0 if scheme == "https" else 0.0,
        "has_ip": float(_is_ip(host_no_port)),
        "len_url": float(len(url)),
        "len_host": float(len(host_no_port)),
        "len_path": float(len(path)),
        "num_dots_host": float(host_no_port.count(".")),
        "num_hyphen_host": float(host_no_port.count("-")),
        "num_digits_url": float(len(re.findall(r"\d", url))),
        "num_query_params": float(len(query.split("&")) if query else 0),
        "has_at": 1.0 if "@" in url else 0.0,
        "has_frag": 1.0 if fragment else 0.0,
        "path_depth": float(len([t for t in path.split("/") if t])),
    }
    return features
