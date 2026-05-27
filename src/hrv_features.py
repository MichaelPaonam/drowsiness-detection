"""
HRV feature extraction pipeline.

Steps:
  1. R-peak detection via NeuroKit2 (Pan-Tompkins)
  2. RR interval computation
  3. Artifact rejection (physiological bounds + relative outlier removal)
  4. Time-domain HRV feature computation
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import neurokit2 as nk

from config import (
    ECG_SAMPLE_RATE,
    RPEAK_METHOD,
    RR_MIN_MS,
    RR_MAX_MS,
    RR_OUTLIER_THRESHOLD,
    MIN_RR_INTERVALS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HRVFeatures:
    """Time-domain HRV feature vector for one analysis window."""
    mean_rr: float   # mean RR interval (ms)
    mean_hr: float   # mean heart rate (bpm)
    sdnn: float      # std of NN intervals (ms)
    rmssd: float     # root mean square of successive differences (ms)
    nn50: int        # count of successive diffs > 50 ms
    pnn50: float     # proportion of NN50 (%)
    cv_rr: float     # coefficient of variation (sdnn / mean_rr)

    def to_feature_dict(self) -> dict:
        """Returns the 7 ML-ready features."""
        return {
            "mean_rr": self.mean_rr,
            "mean_hr": self.mean_hr,
            "sdnn":    self.sdnn,
            "rmssd":   self.rmssd,
            "nn50":    self.nn50,
            "pnn50":   self.pnn50,
            "cv_rr":   self.cv_rr,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_hrv_features(
    ecg_signal: np.ndarray,
    fs: int = ECG_SAMPLE_RATE,
    subject_id=None,
    test_id=None,
) -> Optional[HRVFeatures]:
    """
    Run the full HRV feature extraction pipeline on a filtered ECG signal.

    Returns None if the signal cannot produce valid HRV features (e.g. too few
    clean RR intervals after artifact rejection). Callers should skip None returns.

    Parameters
    ----------
    ecg_signal : np.ndarray
        Pre-filtered ECG signal.
    fs : int
        Sampling frequency (Hz).
    subject_id, test_id : optional
        Used only for logging context.
    """
    ctx = f"s{subject_id}-t{test_id}" if subject_id and test_id else "unknown"

    rpeaks = _detect_rpeaks(ecg_signal, fs, ctx)
    if rpeaks is None:
        return None

    rr_raw   = _compute_rr_ms(rpeaks, fs)
    rr_clean = _reject_artifacts(rr_raw, ctx)

    if len(rr_clean) < MIN_RR_INTERVALS:
        log.warning(
            "[%s] Only %d clean RR intervals (need %d); skipping window.",
            ctx, len(rr_clean), MIN_RR_INTERVALS,
        )
        return None

    features = _compute_features(rr_clean)

    log.debug(
        "[%s] HRV extracted — mean_hr=%.1f bpm, sdnn=%.1f ms, rmssd=%.1f ms, pNN50=%.1f%%",
        ctx, features.mean_hr, features.sdnn, features.rmssd, features.pnn50,
    )
    return features


# ---------------------------------------------------------------------------
# Internal pipeline stages
# ---------------------------------------------------------------------------

def _detect_rpeaks(
    ecg_signal: np.ndarray, fs: int, ctx: str
) -> Optional[np.ndarray]:
    """Detect R-peaks using Pan-Tompkins via NeuroKit2."""
    try:
        _, info = nk.ecg_peaks(ecg_signal, sampling_rate=fs, method=RPEAK_METHOD)
        rpeaks = info["ECG_R_Peaks"]
    except Exception as exc:
        log.error("[%s] R-peak detection failed: %s", ctx, exc)
        return None

    n_peaks       = len(rpeaks)
    duration_min  = len(ecg_signal) / fs / 60
    expected_low  = int(duration_min * 30)
    expected_high = int(duration_min * 200)

    if not (expected_low <= n_peaks <= expected_high):
        log.warning(
            "[%s] Suspicious R-peak count: %d (expected %d-%d for %.1f min recording).",
            ctx, n_peaks, expected_low, expected_high, duration_min,
        )

    return rpeaks


def _compute_rr_ms(rpeaks: np.ndarray, fs: int) -> np.ndarray:
    """Convert R-peak sample indices to RR intervals in milliseconds."""
    return np.diff(rpeaks).astype(float) / fs * 1000.0


def _reject_artifacts(rr_ms: np.ndarray, ctx: str) -> np.ndarray:
    """
    Remove physiologically implausible RR intervals.

    Two passes:
      1. Hard bounds: remove intervals outside [RR_MIN_MS, RR_MAX_MS]
      2. Relative outlier: remove intervals deviating > 20% from the median
    """
    # Pass 1: physiological bounds
    rr_bounded = rr_ms[(rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)]

    if len(rr_bounded) == 0:
        return rr_bounded

    # Pass 2: relative outlier removal
    median_rr = np.median(rr_bounded)
    rr_clean  = rr_bounded[np.abs(rr_bounded - median_rr) <= RR_OUTLIER_THRESHOLD * median_rr]

    removed = len(rr_ms) - len(rr_clean)
    if removed > 0:
        log.debug("[%s] Artifact rejection removed %d/%d RR intervals.",
                  ctx, removed, len(rr_ms))

    return rr_clean


def _compute_features(rr_ms: np.ndarray) -> HRVFeatures:
    """Compute all time-domain HRV features from a clean RR sequence."""
    successive_diffs = np.abs(np.diff(rr_ms))
    mean_rr = float(np.mean(rr_ms))

    nn50  = int(np.sum(successive_diffs > 50.0))
    pnn50 = nn50 / len(successive_diffs) * 100.0 if len(successive_diffs) > 0 else 0.0

    return HRVFeatures(
        mean_rr = mean_rr,
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0,
        sdnn    = float(np.std(rr_ms, ddof=1)),
        rmssd   = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0,
        nn50    = nn50,
        pnn50   = pnn50,
        cv_rr   = float(np.std(rr_ms, ddof=1) / mean_rr) if mean_rr > 0 else 0.0,
    )
