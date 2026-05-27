"""
End-to-end drowsiness detection pipeline runner.

Runs all 6 steps in sequence by importing and calling each module's run()
function directly. Stops immediately if any step fails.

Usage
-----
    python src/run_all.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, str(Path(__file__).parent))

import extract_ecg
import extract_hrv_windows
import generate_pseudo_kss
import normalize_features
import smooth_labels
import train_pseudo_model

log = logging.getLogger(__name__)

STEPS = [
    ("Extracting ECG signals",         lambda: extract_ecg.run(dataset="all")),
    ("Extracting windowed HRV",         lambda: extract_hrv_windows.run(dataset="all")),
    ("Normalizing features",            lambda: normalize_features.run()),
    ("Generating pseudo-KSS labels",    lambda: generate_pseudo_kss.run(method="both")),
    ("Smoothing labels",                lambda: smooth_labels.run()),
    ("Training pseudo-label models",    lambda: train_pseudo_model.run()),
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    setup_logging()

    n_steps = len(STEPS)
    pipeline_start = time.time()

    for i, (label, fn) in enumerate(STEPS, start=1):
        print(f"\n{'='*60}")
        print(f"  Step {i}/{n_steps}: {label}")
        print(f"{'='*60}")
        step_start = time.time()
        try:
            fn()
        except Exception as exc:
            log.error("Step %d/%d FAILED: %s", i, n_steps, exc, exc_info=True)
            print(f"\n[ERROR] Pipeline stopped at step {i}/{n_steps}: {label}")
            sys.exit(1)
        elapsed = time.time() - step_start
        print(f"\n  Step {i}/{n_steps} completed in {elapsed:.1f}s")

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete — total time: {total:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
