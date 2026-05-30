import numpy as np
from sklearn.metrics import roc_curve


def find_threshold_for_fpr(y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float) -> float:
    """
    Find the decision threshold that results in a False Positive Rate (FPR)
    closest to, but not exceeding, the target_fpr.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        target_fpr: Desired False Positive Rate (0.0 to 1.0).

    Returns:
        Optimal probability threshold as a float.
    """
    fpr, _, thresholds = roc_curve(y_true, y_proba)

    # Find indices where FPR <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]

    if len(valid_indices) == 0:
        # Fallback to a very high threshold if target_fpr is extremely low
        return 1.0

    # Return the threshold corresponding to the highest FPR <= target_fpr
    return float(thresholds[valid_indices[-1]])
