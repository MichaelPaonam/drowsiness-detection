"""
Drowsiness detection inference pipeline.

Provides DrowsinessPredictor for classifying ECG signals or pre-computed
HRV features as alert (0) or drowsy (1), and train_and_save_model to
produce a deployment-ready model trained on all available data.

Usage
-----
    python src/inference.py --train
    python src/inference.py --predict-ecg path/to/ecg.csv --sampling-rate 512
    python src/inference.py --predict-features path/to/features.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    HRV_WINDOWS_FINAL_CSV,
    OUTPUTS_DIR,
    SELECTED_FEATURES,
    STEP_SEC,
    WINDOW_SEC,
)
from hrv_extractor import apply_bandpass_filter  # noqa: E402
from hrv_metrics import compute_window_hrv  # noqa: E402

log = logging.getLogger(__name__)

MODELS_DIR = OUTPUTS_DIR / "models"
_LABEL_MAP: dict[int, str] = {0: "alert", 1: "drowsy"}
_RAW_FEATURE_COLS: list[str] = [col.replace("_znorm", "") for col in SELECTED_FEATURES]
_RANDOM_STATE: int = 42


def _build_catboost() -> CatBoostClassifier:
    """Return a class-balanced CatBoostClassifier (unfitted)."""
    return CatBoostClassifier(
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=_RANDOM_STATE,
    )


def _build_xgb(scale_pos_weight: float) -> XGBClassifier:
    """Return an XGBClassifier weighted for class imbalance (unfitted)."""
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=_RANDOM_STATE,
    )


class DrowsinessPredictor:
    """
    End-to-end drowsiness inference from raw ECG or pre-computed HRV features.

    Wraps a trained CatBoost or XGBoost model together with a fitted
    StandardScaler and exposes three prediction surfaces:

    * predict_from_ecg      — full pipeline from a raw ECG array
    * predict_from_features — single-sample prediction from znorm features
    * predict_batch         — batch prediction from a pre-computed feature CSV
    """

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        model_type: str = "xgboost",
    ) -> None:
        """
        Load a trained model and fitted scaler from disk.

        Args:
            model_path: Path to the model file (.cbm for CatBoost, .json for XGBoost).
            scaler_path: Path to the StandardScaler pickle saved by joblib.
            model_type: One of "catboost" or "xgboost".

        Raises:
            ValueError: If model_type is not recognized.
            FileNotFoundError: If model_path or scaler_path does not exist.
        """
        if model_type not in ("catboost", "xgboost"):
            raise ValueError(f"model_type must be 'catboost' or 'xgboost', got '{model_type}'.")

        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_type = model_type

        self.scaler: StandardScaler = joblib.load(scaler_path)
        log.info("Loaded scaler from %s", scaler_path)

        if model_type == "catboost":
            self._model: CatBoostClassifier | XGBClassifier = CatBoostClassifier()
            self._model.load_model(str(model_path))
        else:
            self._model = XGBClassifier()
            self._model.load_model(str(model_path))
        log.info("Loaded %s model from %s", model_type, model_path)

    def predict_from_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: int,
    ) -> list[dict[str, Any]]:
        """
        Run the full ECG → HRV → normalize → predict pipeline.

        Bandpass-filters the signal, slides WINDOW_SEC/STEP_SEC windows,
        extracts time-domain HRV features per window, applies within-recording
        z-score normalization (mimicking per-subject normalization from
        training), applies the saved StandardScaler, and returns per-window
        predictions.

        Args:
            ecg_signal: Raw 1-D ECG signal as a numpy array.
            sampling_rate: Signal sampling rate in Hz.

        Returns:
            List of per-window dicts with keys: window_start_sec,
            window_end_sec, prediction, confidence, and one key per
            SELECTED_FEATURES column containing the znorm value.

        Raises:
            ValueError: If no valid HRV windows can be extracted from the signal.
        """
        ecg_filt = apply_bandpass_filter(ecg_signal.astype(np.float64), float(sampling_rate))

        window_samples = int(WINDOW_SEC * sampling_rate)
        step_samples = int(STEP_SEC * sampling_rate)
        n_samples = len(ecg_filt)

        raw_records: list[dict[str, Any]] = []
        start = 0
        while start + window_samples <= n_samples:
            end = start + window_samples
            hrv = compute_window_hrv(ecg_filt[start:end], fs=sampling_rate)
            if hrv is not None:
                raw_records.append(
                    {
                        "window_start_sec": start / sampling_rate,
                        "window_end_sec": end / sampling_rate,
                        **{col: getattr(hrv, col) for col in _RAW_FEATURE_COLS},
                    }
                )
            start += step_samples

        if not raw_records:
            raise ValueError(
                "No valid HRV windows could be extracted from the ECG signal. "
                "Check that the signal is long enough and has acceptable quality."
            )

        raw_df = pd.DataFrame(raw_records)
        znorm: dict[str, np.ndarray] = {}
        for raw_col, znorm_col in zip(_RAW_FEATURE_COLS, SELECTED_FEATURES):
            vals = raw_df[raw_col].to_numpy(dtype=float)
            mu = float(vals.mean())
            sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            znorm[znorm_col] = (vals - mu) / sigma if sigma > 0.0 else np.zeros_like(vals)

        X = np.column_stack([znorm[col] for col in SELECTED_FEATURES]).astype(np.float32)
        X_scaled = self.scaler.transform(X)
        y_proba: np.ndarray = self._model.predict_proba(X_scaled)[:, 1]

        results: list[dict[str, Any]] = []
        for i, row in enumerate(raw_records):
            confidence = float(y_proba[i])
            result: dict[str, Any] = {
                "window_start_sec": row["window_start_sec"],
                "window_end_sec": row["window_end_sec"],
                "prediction": _LABEL_MAP[int(confidence >= 0.5)],
                "confidence": round(confidence, 4),
            }
            for znorm_col in SELECTED_FEATURES:
                result[znorm_col] = round(float(znorm[znorm_col][i]), 6)
            results.append(result)

        n_drowsy = sum(1 for r in results if r["prediction"] == "drowsy")
        log.info(
            "Processed %d windows: %d drowsy, %d alert",
            len(results),
            n_drowsy,
            len(results) - n_drowsy,
        )
        return results

    def predict_from_features(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Predict drowsiness from a single set of pre-computed znorm HRV features.

        Args:
            features: Mapping of SELECTED_FEATURES column names (e.g. "sdnn_znorm")
                to their scalar values. Extra keys are silently ignored.

        Returns:
            Dict with keys "prediction" ("alert" or "drowsy") and "confidence"
            (probability of drowsy in [0, 1]).

        Raises:
            KeyError: If any SELECTED_FEATURES column is absent from features.
        """
        missing = [col for col in SELECTED_FEATURES if col not in features]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")

        X = np.array([[features[col] for col in SELECTED_FEATURES]], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        confidence = float(self._model.predict_proba(X_scaled)[0, 1])
        return {
            "prediction": _LABEL_MAP[int(confidence >= 0.5)],
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, csv_path: Path) -> pd.DataFrame:
        """
        Load a CSV of pre-computed HRV features and predict for every row.

        The CSV must contain all SELECTED_FEATURES columns. Any additional
        columns are preserved in the returned DataFrame.

        Args:
            csv_path: Path to the CSV file with per-window HRV feature rows.

        Returns:
            DataFrame with the original columns plus "prediction" and "confidence".

        Raises:
            FileNotFoundError: If csv_path does not exist.
            KeyError: If any SELECTED_FEATURES column is missing from the CSV.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        missing = [col for col in SELECTED_FEATURES if col not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns in CSV: {missing}")

        X = df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        y_proba: np.ndarray = self._model.predict_proba(X_scaled)[:, 1]

        out = df.copy()
        out["prediction"] = [_LABEL_MAP[int(p >= 0.5)] for p in y_proba]
        out["confidence"] = y_proba
        log.info("Predicted %d rows from %s", len(out), csv_path)
        return out


def train_and_save_model(
    model_type: str = "xgboost",
    csv_path: Path | None = None,
    models_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Train a model on all available labeled data and save it with its scaler.

    Unlike train_classifiers.py (which evaluates per-fold for reporting),
    this function trains on the complete dataset to produce a
    deployment-ready model suitable for use with DrowsinessPredictor.

    Args:
        model_type: Model to train; one of "catboost" (default) or "xgboost".
        csv_path: Path to the labeled HRV feature CSV.
            Defaults to HRV_WINDOWS_FINAL_CSV from config.
        models_dir: Output directory for saved files.
            Defaults to OUTPUTS_DIR/models.

    Returns:
        Tuple of (model_path, scaler_path).

    Raises:
        ValueError: If model_type is not recognized or dataset has no positive labels.
        FileNotFoundError: If csv_path does not exist.
        KeyError: If required columns are absent from the dataset.
    """
    if model_type not in ("catboost", "xgboost"):
        raise ValueError(f"model_type must be 'catboost' or 'xgboost', got '{model_type}'.")

    csv_path = Path(csv_path) if csv_path is not None else HRV_WINDOWS_FINAL_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run the full data pipeline first."
        )

    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    log.info("Loaded %d windows from %s", len(df), csv_path)

    required_cols = SELECTED_FEATURES + ["pseudo_label_smoothed"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    X = df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
    y = df["pseudo_label_smoothed"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info("Fitted StandardScaler on %d samples", len(X))

    n_pos = int(y.sum())
    if model_type == "catboost":
        model: CatBoostClassifier | XGBClassifier = _build_catboost()
        model_path = models_dir / "best_model.cbm"
    else:
        if n_pos == 0:
            raise ValueError("Dataset has no positive samples; cannot compute scale_pos_weight.")
        model = _build_xgb((len(y) - n_pos) / n_pos)
        model_path = models_dir / "best_model.json"

    log.info(
        "Training %s on %d samples (%d positive, %d negative)...",
        model_type,
        len(X),
        n_pos,
        len(y) - n_pos,
    )
    model.fit(X_scaled, y)
    log.info("Training complete.")

    model.save_model(str(model_path))
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    log.info("Model  -> %s", model_path)
    log.info("Scaler -> %s", scaler_path)
    return model_path, scaler_path


def setup_logging() -> None:
    """Configure logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drowsiness detection inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train",
        action="store_true",
        help="Train the best model on all data and save to outputs/models/",
    )
    mode.add_argument(
        "--predict-ecg",
        metavar="ECG_CSV",
        type=Path,
        help="Path to ECG CSV (columns: timestamp_sec, ecg)",
    )
    mode.add_argument(
        "--predict-features",
        metavar="FEATURES_CSV",
        type=Path,
        help="Path to pre-computed HRV feature CSV (must contain SELECTED_FEATURES columns)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=512,
        metavar="HZ",
        help="ECG sampling rate in Hz (only used with --predict-ecg)",
    )
    parser.add_argument(
        "--model-type",
        choices=["catboost", "xgboost"],
        default="xgboost",
        help="Model architecture",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to saved model file (default: outputs/models/best_model.*)",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to saved scaler pkl (default: outputs/models/scaler.pkl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV path for predictions (default: print to stdout)",
    )
    args = parser.parse_args()
    setup_logging()

    if args.train:
        saved_model, saved_scaler = train_and_save_model(model_type=args.model_type)
        print(f"Model  saved: {saved_model}")
        print(f"Scaler saved: {saved_scaler}")
        sys.exit(0)

    model_path = args.model_path
    if model_path is None:
        ext = "cbm" if args.model_type == "catboost" else "json"
        model_path = MODELS_DIR / f"best_model.{ext}"

    scaler_path = args.scaler_path or (MODELS_DIR / "scaler.pkl")

    predictor = DrowsinessPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        model_type=args.model_type,
    )

    if args.predict_ecg:
        ecg_df = pd.read_csv(args.predict_ecg)
        if "ecg" not in ecg_df.columns:
            log.error(
                "ECG CSV must contain an 'ecg' column. Found: %s",
                list(ecg_df.columns),
            )
            sys.exit(1)
        ecg_signal = ecg_df["ecg"].to_numpy(dtype=np.float64)
        window_results = predictor.predict_from_ecg(ecg_signal, sampling_rate=args.sampling_rate)
        out_df = pd.DataFrame(window_results)
    else:
        out_df = predictor.predict_batch(args.predict_features)

    if args.output:
        out_df.to_csv(args.output, index=False)
        log.info("Predictions saved -> %s", args.output)
    else:
        sys.stdout.write(out_df.to_csv(index=False))
