from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def binary_metrics(y_true, y_prob, threshold=0.5):
    auc = roc_auc_score(y_true, y_prob)
    preds = (y_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary')
    return {'auc': auc, 'precision': p, 'recall': r, 'f1': f}
