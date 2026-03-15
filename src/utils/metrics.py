import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    precision_score,
    confusion_matrix,
)


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["auprc"] = None

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision_abnormal"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall_abnormal"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if (precision_normal + recall_normal) > 0 else 0.0
    )

    precision_abnormal = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_abnormal = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_abnormal = (
        2 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal)
        if (precision_abnormal + recall_abnormal) > 0 else 0.0
    )

    metrics["normal"] = {
        "acc": recall_normal,
        "precision": precision_normal,
        "recall": recall_normal,
        "f1": f1_normal,
        "support": int((y_true == 0).sum()),
    }

    metrics["abnormal"] = {
        "acc": recall_abnormal,
        "precision": precision_abnormal,
        "recall": recall_abnormal,
        "f1": f1_abnormal,
        "support": int((y_true == 1).sum()),
    }

    return metrics


def search_best_threshold(y_true, y_prob, metric="f1"):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    best_threshold = 0.5
    best_score = -1
    best_stats = {}

    thresholds = np.arange(0.01, 1.00, 0.01)

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if metric == "f1":
            score = f1
        elif metric == "recall":
            score = recall
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = th
            best_stats = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    return best_threshold, best_score, best_stats