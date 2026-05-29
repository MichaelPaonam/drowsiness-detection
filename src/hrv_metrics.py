"""
HRV feature extraction pipeline.

Provides functionality for R-peak detection, artifact rejection, and
time-domain HRV feature computation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import neurokit2 as nk
import numpy as np

from config import (
    ECG_SAMPLE_RATE,
    MIN_RR_INTERVALS,
    RPEAK_METHOD,
    RR_MAX_MS,
    RR_MIN_MS,
    RR_OUTLIER_THRESHOLD,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HRVFeatures:
    """Time-domain HRV feature vector for an analysis window."""

    mean_rr: float
    mean_hr: float
    sdnn: float
    rmssd: float
    nn50: int
    pnn50: float
    cv_rr: float

    def to_dict(self) -> dict[str, float | int]:
        """Returns the ML-ready feature dictionary."""
        return {
            "mean_rr": self.mean_rr,
            "mean_hr": self.mean_hr,
            "sdnn": self.sdnn,
            "rmssd": self.rmssd,
            "nn50": self.nn50,
            "pnn50": self.pnn50,
            "cv_rr": self.cv_rr,
        }


def compute_window_hrv(
    ecg_signal: np.ndarray,
    fs: int = ECG_SAMPLE_RATE,
    subject_id: Optional[str] = None,
    trial_id: Optional[int] = None,
) -> Optional[HRVFeatures]:
    """
    Extracts time-domain HRV features from a pre-filtered ECG signal window.

    Returns None if valid features cannot be computed (e.g., insufficient clean RR intervals).
    """
    ctx = f"s{subject_id}-t{trial_id}" if subject_id and trial_id else "unknown"

    # 1. R-peak detection
    try:
        _, info = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method=RPEAK_METHOD)
        rpeaks = info["ECG_R_Peaks"]
    except (ValueError, KeyError, TypeError) as exc:
        log.error("[%s] R-peak detection failed: %s", ctx, exc)
        return None

    # 2. Compute RR intervals (ms)
    if len(rpeaks) < 2:
        return None
    rr_ms = np.diff(rpeaks).astype(float) / fs * 1000.0

    # 3. Artifact rejection
    # Pass 1: Physiological bounds
    rr_clean = rr_ms[(rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)]

    if len(rr_clean) == 0:
        return None

    # Pass 2: Relative outlier removal
    median_rr = np.median(rr_clean)
    rr_clean = rr_clean[np.abs(rr_clean - median_rr) <= RR_OUTLIER_THRESHOLD * median_rr]

    # 4. Validate interval count
    if len(rr_clean) < MIN_RR_INTERVALS:
        log.debug("[%s] Insufficient clean intervals (%d)", ctx, len(rr_clean))
        return None

    return _calculate_time_domain_features(rr_clean)


def _calculate_time_domain_features(rr_ms: np.ndarray) -> HRVFeatures:
    """Computes time-domain features from clean RR intervals."""
    successive_diffs = np.abs(np.diff(rr_ms))
    mean_rr = float(np.mean(rr_ms))

    nn50 = int(np.sum(successive_diffs > 50.0))
    pnn50 = nn50 / len(successive_diffs) * 100.0 if len(successive_diffs) > 0 else 0.0

    return HRVFeatures(
        mean_rr=mean_rr,
        mean_hr=60000.0 / mean_rr if mean_rr > 0 else 0.0,
        sdnn=float(np.std(rr_ms, ddof=1)),
        rmssd=float(np.sqrt(np.mean(successive_diffs**2))) if len(successive_diffs) > 0 else 0.0,
        nn50=nn50,
        pnn50=pnn50,
        cv_rr=float(np.std(rr_ms, ddof=1) / mean_rr) if mean_rr > 0 else 0.0,
    )
