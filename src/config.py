"""Central configuration for the drowsiness detection pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Root of the project (one level above src/)
PROJECT_ROOT = Path(__file__).parent.parent


def _env_path(var: str, default: Path) -> Path:
    raw = os.getenv(var)
    if raw is None:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ── Dataset paths (read-only) ─────────────────────────────────────────────────

DROZY_ROOT = _env_path("DROZY_ROOT", PROJECT_ROOT / "data" / "raw" / "DROZY")
DDD_DIR    = _env_path("DDD_DIR",    PROJECT_ROOT / "data" / "raw" / "DDD")

PSG_DIR  = DROZY_ROOT / "psg"
KSS_FILE = DROZY_ROOT / "KSS.txt"

# ── Output directories ────────────────────────────────────────────────────────

OUTPUTS_DIR    = _env_path("OUTPUTS_DIR",   PROJECT_ROOT / "outputs")
ECG_CSV_DIR    = _env_path("ECG_CSV_DIR",   PROJECT_ROOT / "data" / "preprocessed" / "ecg_csv")
PSEUDO_KSS_DIR = OUTPUTS_DIR / "pseudo_kss"

OUTPUTS_DIR.mkdir(exist_ok=True)

# ── ECG acquisition parameters ───────────────────────────────────────────────

ECG_SAMPLE_RATE = 512   # Hz (DROZY); DDD uses 128 Hz — passed explicitly

# Bandpass filter
FILTER_LOWCUT  = 0.5    # Hz
FILTER_HIGHCUT = 40.0   # Hz
FILTER_ORDER   = 4

# R-peak detection
RPEAK_METHOD = "pantompkins1985"

# RR artifact rejection
RR_MIN_MS            = 300    # ~200 bpm upper HR limit
RR_MAX_MS            = 2000   # ~30 bpm lower HR limit
RR_OUTLIER_THRESHOLD = 0.20   # fraction of median; removes ectopic beats

# Minimum clean RR intervals required to compute HRV features in a window
MIN_RR_INTERVALS = 50

# ── Windowing parameters ──────────────────────────────────────────────────────

WINDOW_SEC = 300   # 5-minute analysis window
STEP_SEC   = 60    # 1-minute step (80% overlap)

# ── Clustering ────────────────────────────────────────────────────────────────

N_CLUSTERS = 3

# ── KSS label rule ────────────────────────────────────────────────────────────

KSS_DROWSY_THRESHOLD = 7   # KSS >= this → Drowsy (1)

# Tests missing due to backup loss or never occurring
EXCLUDED_TESTS = {
    (7, 1),
    (9, 1),
    (10, 2),
    (12, 2),
    (12, 3),
    (13, 3),
}

# Subject-level fixed split (used by split_utils.py for DROZY ground-truth lookup)
SPLIT_MAP = {
    "train": [1, 2, 4, 5, 7, 8, 9, 10, 13],
    "val":   [3, 6, 11],
    "test":  [12, 14],
}

# ── HRV feature column names (in order) ──────────────────────────────────────

HRV_FEATURE_COLS = [
    "mean_rr",
    "mean_hr",
    "sdnn",
    "rmssd",
    "nn50",
    "pnn50",
    "cv_rr",
]

# ── Pipeline output file paths ────────────────────────────────────────────────

HRV_WINDOWS_CSV            = OUTPUTS_DIR / "hrv_windows.csv"
HRV_WINDOWS_NORMALIZED_CSV = OUTPUTS_DIR / "hrv_windows_normalized.csv"
HRV_WINDOWS_PSEUDO_CSV     = OUTPUTS_DIR / "hrv_windows_pseudo_labeled.csv"
HRV_WINDOWS_FINAL_CSV      = OUTPUTS_DIR / "hrv_windows_final.csv"
PSEUDO_MODEL_COMPARISON_TXT = OUTPUTS_DIR / "pseudo_model_comparison.txt"
PSEUDO_MODEL_RESULTS_CSV   = OUTPUTS_DIR / "pseudo_model_results.csv"
