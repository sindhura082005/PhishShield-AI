import pandas as pd
import re
import tldextract
import math

def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    from collections import Counter
    if not s:
        return 0
    probs = [count / len(s) for count in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs)

def preprocess_urls(raw_csv, output_csv):
    df = pd.read_csv(raw_csv)

    # Feature engineering
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].str.count(r'\.')
    df['num_hyphens'] = df['url'].str.count(r'-')
    df['num_digits'] = df['url'].str.count(r'\d')
    df['num_special_chars'] = df['url'].str.count(r'[^\w]')
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith("https") else 0)
    df['is_ip_address'] = df['url'].apply(lambda x: 1 if re.match(r'^https?:\/\/\d+\.\d+\.\d+\.\d+', x) else 0)
    
    
    df['tld'] = df['url'].apply(lambda x: tldextract.extract(x).suffix)
    
    df['entropy'] = df['url'].apply(lambda x: shannon_entropy(tldextract.extract(x).domain))
    
    df['encoded_label'] = df['label'].apply(lambda x: 1 if x.lower() == "phish" else 0)

    df.to_csv(output_csv, index=False)
    return df

processed_df = preprocess_urls("Phishing_URL_Dataset.csv", "processed_phishing_urls.csv")