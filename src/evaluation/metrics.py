from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_anomaly_detection(y_true, y_score):
    """
    Compute AUC-ROC and AUC-PR for anomaly detection.
    """
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    return {
        "AUC_ROC": round(auc_roc, 4),
        "AUC_PR": round(auc_pr, 4)
    }
