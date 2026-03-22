import numpy as np
import pandas as pd

from src.utils.metrics import compute_score_based_metrics, search_best_threshold


def calibrate_scores_casewise(scores, case_ids, method="zscore", eps=1e-6):
    """
    Case-wise score normalization.

    Args:
        scores: list/np.ndarray, raw anomaly scores, higher => more abnormal
        case_ids: list/np.ndarray, same length as scores
        method:
            - "zscore": (x - mean_case) / std_case
            - "robust_zscore": (x - median_case) / (1.4826 * MAD)
            - "percentile": within-case rank percentile in [0, 1]
        eps: small constant

    Returns:
        calibrated_scores: np.ndarray
    """
    scores = np.asarray(scores, dtype=np.float64)
    case_ids = np.asarray(case_ids)

    if scores.shape[0] != case_ids.shape[0]:
        raise ValueError("scores and case_ids must have the same length")

    calibrated = np.zeros_like(scores, dtype=np.float64)

    unique_cases = pd.unique(case_ids)

    for cid in unique_cases:
        mask = case_ids == cid
        case_scores = scores[mask]

        if case_scores.size == 0:
            continue

        if method == "zscore":
            mu = case_scores.mean()
            sigma = case_scores.std(ddof=0)
            calibrated[mask] = (case_scores - mu) / (sigma + eps)

        elif method == "robust_zscore":
            med = np.median(case_scores)
            mad = np.median(np.abs(case_scores - med))
            scale = 1.4826 * mad + eps
            calibrated[mask] = (case_scores - med) / scale

        elif method == "percentile":
            if case_scores.size == 1:
                calibrated[mask] = 0.5
            else:
                order = np.argsort(case_scores)
                ranks = np.empty_like(order, dtype=np.float64)
                ranks[order] = np.arange(case_scores.size, dtype=np.float64)
                calibrated[mask] = ranks / (case_scores.size - 1)

        else:
            raise ValueError(f"Unsupported case-wise calibration method: {method}")

    return calibrated


def evaluate_casewise_calibration(y_true, raw_scores, case_ids, threshold=0.5, method="zscore"):
    calibrated_scores = calibrate_scores_casewise(
        scores=raw_scores,
        case_ids=case_ids,
        method=method,
    )
    metrics = compute_score_based_metrics(
        y_true=y_true,
        y_score=calibrated_scores,
        threshold=threshold,
        higher_score_more_positive=True,
    )
    return metrics, calibrated_scores


def search_best_threshold_casewise(y_true, raw_scores, case_ids, metric="f1", method="zscore"):
    calibrated_scores = calibrate_scores_casewise(
        scores=raw_scores,
        case_ids=case_ids,
        method=method,
    )
    best_th, best_score, best_stats = search_best_threshold(
        y_true=y_true,
        y_score=calibrated_scores,
        metric=metric,
        higher_score_more_positive=True,
    )
    return best_th, best_score, best_stats, calibrated_scores


def summarize_case_isolation(train_csv, val_csv, test_csv):
    """
    Check whether train / val / test are case-isolated.
    Returns a dict that you can print or save.
    """
    result = {
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
    }

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "case_id" not in df.columns:
            result[f"{name}_has_case_id"] = False
            result[f"{name}_num_cases"] = None
        else:
            result[f"{name}_has_case_id"] = True
            result[f"{name}_num_cases"] = int(df["case_id"].nunique())

    if not all("case_id" in df.columns for df in [train_df, val_df, test_df]):
        result["case_isolation_checkable"] = False
        result["train_val_overlap_cases"] = None
        result["train_test_overlap_cases"] = None
        result["val_test_overlap_cases"] = None
        result["is_case_isolated"] = None
        return result

    train_cases = set(train_df["case_id"].astype(str).unique().tolist())
    val_cases = set(val_df["case_id"].astype(str).unique().tolist())
    test_cases = set(test_df["case_id"].astype(str).unique().tolist())

    train_val_overlap = sorted(train_cases & val_cases)
    train_test_overlap = sorted(train_cases & test_cases)
    val_test_overlap = sorted(val_cases & test_cases)

    result["case_isolation_checkable"] = True
    result["train_val_overlap_cases"] = train_val_overlap
    result["train_test_overlap_cases"] = train_test_overlap
    result["val_test_overlap_cases"] = val_test_overlap
    result["num_train_val_overlap"] = len(train_val_overlap)
    result["num_train_test_overlap"] = len(train_test_overlap)
    result["num_val_test_overlap"] = len(val_test_overlap)
    result["is_case_isolated"] = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )
    return result