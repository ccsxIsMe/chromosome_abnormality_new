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


def _compute_binary_metrics_from_scores(y_true, y_score, threshold=0.5, higher_score_more_positive=True):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if higher_score_more_positive:
        y_pred = (y_score >= threshold).astype(int)
        pos_score = y_score
    else:
        y_pred = (y_score <= threshold).astype(int)
        pos_score = -y_score

    metrics = {}

    try:
        metrics["auroc"] = roc_auc_score(y_true, pos_score)
    except Exception:
        metrics["auroc"] = None

    try:
        metrics["auprc"] = average_precision_score(y_true, pos_score)
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


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    return _compute_binary_metrics_from_scores(
        y_true=y_true,
        y_score=y_prob,
        threshold=threshold,
        higher_score_more_positive=True,
    )


def compute_score_based_metrics(y_true, y_score, threshold=0.5, higher_score_more_positive=True):
    return _compute_binary_metrics_from_scores(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
        higher_score_more_positive=higher_score_more_positive,
    )


def _build_threshold_candidates(y_score, max_points=200):
    y_score = np.array(y_score, dtype=np.float64)
    y_score = y_score[np.isfinite(y_score)]

    if y_score.size == 0:
        return np.array([0.5], dtype=np.float64)

    unique_scores = np.unique(y_score)
    if unique_scores.size <= max_points:
        return unique_scores

    lo = float(unique_scores.min())
    hi = float(unique_scores.max())

    if lo == hi:
        return np.array([lo], dtype=np.float64)

    return np.linspace(lo, hi, num=max_points)


def search_best_threshold(y_true, y_score, metric="f1", higher_score_more_positive=True):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    best_threshold = 0.5
    best_score = -1
    best_stats = {}

    thresholds = _build_threshold_candidates(y_score)

    for th in thresholds:
        if higher_score_more_positive:
            y_pred = (y_score >= th).astype(int)
            auc_score_input = y_score
        else:
            y_pred = (y_score <= th).astype(int)
            auc_score_input = -y_score

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        if metric == "f1":
            score = f1
        elif metric == "recall":
            score = recall
        elif metric == "balanced_acc":
            score = bal_acc
        elif metric == "auprc":
            try:
                score = average_precision_score(y_true, auc_score_input)
            except Exception:
                score = -1
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(th)
            best_stats = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "balanced_acc": float(bal_acc),
            }

    return best_threshold, best_score, best_stats