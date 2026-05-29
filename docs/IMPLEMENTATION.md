# Drowsiness Detection - Implementation Documentation

## Project Overview

**Drowsiness Detection** is a data science system designed to identify signs of fatigue from physiological signals. The current implementation focuses on **ECG (electrocardiogram)-based drowsiness detection using Heart Rate Variability (HRV) analysis**.

### Primary Objectives
- Detect drowsiness from ECG signals using HRV features
- Build a supervised learning pipeline with pseudo-labeled training data
- Classify drowsiness into three states: **alert**, **transitional**, **drowsy**
- Support subject-independent model validation to avoid data leakage

### Target Domains
- Driver monitoring and fatigue detection
- Occupational safety applications
- Research on physiological indicators of drowsiness

---

## Project Structure

```
drowsiness-detection/
├── data/
│   ├── raw/                      # Raw dataset directories
│   │   ├── DROZY/               # DROZY dataset (EDF files)
│   │   │   ├── psg/             # PSG signals in EDF format
│   │   │   └── KSS.txt          # Karolinska Sleepiness Scale labels
│   │   └── DDD/                 # DDD (Driver Drowsiness Detection) dataset
│   └── preprocessed/
│       └── ecg_csv/             # Extracted ECG signals as CSV
│           ├── drozy/
│           └── ddd/
├── outputs/                      # Generated outputs and results
│   ├── hrv_windows.csv          # Raw HRV feature windows
│   ├── hrv_windows_normalized.csv # Per-subject z-score normalized HRV
│   ├── hrv_windows_final.csv    # Smoothed and finalized labels
│   ├── pseudo_kss/              # Pseudo-label generation outputs
│   │   ├── fig_cluster_scatter.png
│   │   ├── fig_cluster_distribution.png
│   │   ├── fig_cluster_centroids.png
│   │   ├── fig_silhouette.png
│   │   └── fig_temporal_smoothing.png
├── src/                          # Main source code
│   ├── config.py                # Central configuration
│   ├── extract_ecg.py           # ECG extraction from EDF files
│   ├── hrv_extractor.py         # HRV feature computation
│   ├── hrv_metrics.py           # HRV metrics and R-peak detection
│   ├── generate_pseudo_kss.py   # Pseudo-label clustering
│   ├── normalize_features.py    # Per-subject feature normalization
│   ├── smooth_labels.py         # Temporal label smoothing
│   └── split_utils.py           # Train/val/test split utilities
├── docs/                         # Documentation
├── pyproject.toml               # Project metadata and tooling
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── plan.md                      # Detailed project plan
├── CODE_OF_CONDUCT.md           # Community guidelines
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # Project license
└── SECURITY.md                  # Security policy
```

---

## Implemented Modules

### 1. **config.py** - Central Configuration
Centralized configuration for the entire pipeline.

**Key Features:**
- Environment variable loading via `.env` file support
- Configurable dataset paths (DROZY and DDD)
- ECG acquisition parameters (sample rate, filter settings)
- Output directory management
- HRV windowing parameters (window size, step size)
- RR-interval artifact rejection thresholds
- Feature column definitions and split mappings

**Usage:**
```python
from src.config import ECG_SAMPLE_RATE, WINDOW_SEC, STEP_SEC
```

**Key Configuration Variables:**
- `ECG_SAMPLE_RATE`: 512 Hz (DROZY), 128 Hz (DDD)
- `FILTER_LOWCUT / FILTER_HIGHCUT`: 0.5–40 Hz bandpass filter
- `WINDOW_SEC`: Window length for HRV computation
- `STEP_SEC`: Stride between windows
- `RPEAK_METHOD`: "pantompkins1985" for R-peak detection

---

### 2. **extract_ecg.py** - ECG Signal Extraction
Extracts raw ECG signals from EDF files to CSV format.

**Purpose:**
Convert binary EDF files to text-based CSV for easier processing and inspection.

**Input:**
- EDF files from DROZY (`data/raw/DROZY/psg/*.edf`)
- EDF files from DDD dataset

**Output:**
- CSV files with columns: `timestamp_sec`, `ecg`
- Path: `data/preprocessed/ecg_csv/{drozy|ddd}/`

**Features:**
- Flexible ECG channel detection (exact "ECG" or "EKG", case-insensitive fallback)
- Transparent handling of both DROZY and DDD datasets
- Dry-run mode for validation before file writes
- Detailed logging of extraction progress

**Usage:**
```bash
python src/extract_ecg.py
python src/extract_ecg.py --dataset drozy
python src/extract_ecg.py --dataset ddd --dry-run
```

---

### 3. **hrv_metrics.py** - HRV Feature Computation
Computes time-domain HRV features from R-R intervals.

**Core Features:**
- **R-peak detection** using NeuroKit2 (pantompkins1985 algorithm)
- **RR-interval artifact rejection**:
  - Physiological bounds: 300–2000 ms
  - Outlier detection: removes RR intervals > 2 std from median
  - Minimum required RR intervals per window

**HRV Features Extracted:**
- `mean_rr`: Mean RR interval (ms)
- `mean_hr`: Mean heart rate (bpm)
- `sdnn`: Standard deviation of RR intervals (ms) — marker of overall HRV
- `rmssd`: Root mean square of successive RR differences (ms) — parasympathetic activity
- `nn50`: Count of RR differences > 50 ms
- `pnn50`: Percentage of RR differences > 50 ms (0–100)
- `cv_rr`: Coefficient of variation (std/mean × 100%)

**Data Structure:**
```python
@dataclass
class HRVFeatures:
    mean_rr: float
    mean_hr: float
    sdnn: float
    rmssd: float
    nn50: int
    pnn50: float
    cv_rr: float
```

**Drowsiness Indicators:**
- ↓ `sdnn` (lower HRV)
- ↓ `rmssd` (reduced parasympathetic tone)
- ↑ `mean_hr` (elevated resting heart rate)
- ↓ `pnn50` (fewer large RR variations)

---

### 4. **hrv_extractor.py** - Windowed HRV Extraction Pipeline
Processes ECG CSV files into windowed HRV features.

**Purpose:**
Extract time-domain HRV features over sliding windows from continuous ECG signals.

**Input:**
- ECG CSV files from `data/preprocessed/ecg_csv/`

**Output:**
- `outputs/hrv_windows.csv`

**Columns in Output:**
- `subject_id`: Unified identifier (e.g., `drozy_s01`, `ddd_01M`)
- `trial_id`: Trial number
- `window_idx`: Sequential window index
- `timestamp_start`: Window start time (seconds)
- `timestamp_end`: Window end time (seconds)
- HRV features: `mean_rr`, `mean_hr`, `sdnn`, `rmssd`, `nn50`, `pnn50`, `cv_rr`
- `rr_count`: Number of RR intervals in the window
- `artifact_count`: Number of rejected RR intervals

**Parameters:**
- `WINDOW_SEC`: Window length (typically 60 seconds)
- `STEP_SEC`: Stride between windows (typically 20 seconds)

**Windowing Strategy:**
- Sliding windows with overlap
- Overlapping windows provide more training samples and smoother temporal coverage

**Usage:**
```bash
python src/hrv_extractor.py
python src/hrv_extractor.py --dataset all
```

---

### 5. **generate_pseudo_kss.py** - Pseudo-Label Generation via Clustering
Generates pseudo-drowsiness labels using unsupervised clustering on normalized HRV features.

**Purpose:**
Create training labels from unlabeled HRV data using domain knowledge about HRV changes during drowsiness.

**Input:**
- `outputs/hrv_windows_normalized.csv` (z-score normalized HRV features)

**Output:**
- `outputs/hrv_windows_pseudo_labeled.csv`
- Visualizations in `outputs/pseudo_kss/`:
  - `fig_cluster_scatter.png`: 2D PCA scatter of clusters
  - `fig_cluster_distribution.png`: Distribution of pseudo-KSS values
  - `fig_cluster_centroids.png`: Feature means by cluster
  - `fig_silhouette.png`: Silhouette analysis

**Clustering Algorithm:**
- **KMeans** (default) or **GMM** (Gaussian Mixture Model)
- **Number of clusters**: 3 (alert, transitional, drowsy)

**Pseudo-Label Mapping:**
Clusters are ranked by mean `sdnn_znorm` (descending):

| Cluster | Mean SDNN | Pseudo-KSS | Label |
|---------|-----------|-----------|-------|
| Highest | High HRV | 2 | Alert |
| Middle | Medium HRV | 5 | Transitional |
| Lowest | Low HRV | 8 | Drowsy |

**Domain Knowledge Applied:**
- Drowsiness ↔ decreased HRV (lower SDNN, RMSSD)
- Drowsiness ↔ slower, more regular heart rate
- Alert states ↔ higher heart rate variability

**Validation (DROZY only):**
- Compares pseudo-KSS clusters to real session-level KSS labels
- Reports accuracy, Cohen's kappa, and confusion matrix

**Usage:**
```bash
python src/generate_pseudo_kss.py
python src/generate_pseudo_kss.py --method kmeans --n-clusters 3
python src/generate_pseudo_kss.py --exclude-subjects drozy_s04
```

---

### 6. **normalize_features.py** - Per-Subject Feature Normalization
Normalizes HRV features within each subject to remove inter-subject baseline differences.

**Purpose:**
Standardize HRV features relative to each subject's own distribution, ensuring clustering reflects **within-subject drowsiness changes** rather than between-subject physiological variation.

**Input:**
- `outputs/hrv_windows.csv` (raw HRV features)

**Output:**
- `outputs/hrv_windows_normalized.csv` (raw features + z-norm columns)

**Normalization Method:**
For each subject and each HRV feature:
```
z_norm = (x - subject_mean) / subject_std
```

**Output Columns:**
- All raw HRV feature columns
- New columns with `_znorm` suffix:
  - `mean_rr_znorm`
  - `mean_hr_znorm`
  - `sdnn_znorm`
  - `rmssd_znorm`
  - etc.

**Edge Cases:**
- Subjects with constant signal (std == 0) → `_znorm = 0` with warning logged

**Usage:**
```bash
python src/normalize_features.py
python src/normalize_features.py --exclude-subjects drozy_s04
```

**Why This Matters:**
- Subject A naturally has HR 60 bpm (alert) → 65 bpm (drowsy)
- Subject B naturally has HR 75 bpm (alert) → 78 bpm (drowsy)
- Unnormalized features make Subject B always look "more alert"
- Z-score normalization corrects for this baseline shift

---

### 7. **smooth_labels.py** - Temporal Label Smoothing
Applies temporal filtering to pseudo-KSS labels to remove noisy isolated flips.

**Purpose:**
Reduce label noise from instantaneous HRV fluctuations by enforcing temporal consistency.

**Input:**
- `outputs/hrv_windows_pseudo_labeled.csv` (clustered labels)

**Output:**
- `outputs/hrv_windows_final.csv` (smoothed labels)
- Visualization: `outputs/pseudo_kss/fig_temporal_smoothing.png`

**Smoothing Methods:**

1. **Median Filter** (default):
   - Applied independently within each subject + trial
   - `kernel_size` parameter (default: 3)
   - Removes isolated label spikes

2. **Enforce Monotonic** (optional):
   - Cumulative maximum within each session
   - Reflects typical design: subjects become **progressively more drowsy**
   - Prevents unrealistic "recovery" mid-session

**Output Columns:**
- All prior columns from pseudo-labeled data
- `pseudo_kss_smoothed`: Post-filtering pseudo-KSS
- `pseudo_label_smoothed`: Binary drowsy indicator (1 if pseudo_kss ≥ 7, else 0)

**Usage:**
```bash
python src/smooth_labels.py
python src/smooth_labels.py --kernel-size 5 --enforce-monotonic
python src/smooth_labels.py --method gmm
```

---

### 8. **split_utils.py** - Train/Val/Test Split Utilities
Provides utilities for subject-level train/validation/test splits.

**Purpose:**
Ensure no subject appears in multiple splits to guarantee independent evaluation and prevent data leakage.

**Core Functions:**

1. **`load_kss()`**
   - Parses `KSS.txt` (session-level labels from DROZY)
   - Returns DataFrame with columns:
     - `subject_id`, `test_id`, `kss_raw`, `label`, `severe_flag`, `excluded`
   - Applies exclusion rules (KSS == 0, missing trials)

2. **Subject-Level Splitting**
   - Respects fixed `SPLIT_MAP` defined in config
   - Guarantees reproducibility
   - Supports subject exclusion (e.g., `--exclude-subjects drozy_s04`)

**KSS Label Mapping:**
- Session-level KSS (1–9 scale):
  - 1–3: Alert
  - 4–6: Transitional
  - 7–9: Drowsy

**Usage:**
```python
from src.split_utils import load_kss

kss_df = load_kss()
train_subjects = split_utils.get_train_subjects()
val_subjects = split_utils.get_val_subjects()
test_subjects = split_utils.get_test_subjects()
```

---

## Data Pipeline & Workflow

### End-to-End Processing Flow

```
1. Raw EDF Files (data/raw/)
   ↓ extract_ecg.py
2. ECG CSV (data/preprocessed/ecg_csv/)
   ↓ hrv_extractor.py
3. Raw HRV Windows (outputs/hrv_windows.csv)
   ↓ normalize_features.py
4. Normalized HRV (outputs/hrv_windows_normalized.csv)
   ↓ generate_pseudo_kss.py
5. Pseudo-Labeled Data (outputs/hrv_windows_pseudo_labeled.csv)
   ↓ smooth_labels.py
6. Final Training Data (outputs/hrv_windows_final.csv)
   ↓ [Model Training & Evaluation]
7. Trained Model & Results
```

### Key Outputs

| File | Purpose | Rows | Columns |
|------|---------|------|---------|
| `hrv_windows.csv` | Raw windowed HRV features | ~1000–10000 | subject, trial, window index, HRV features (7), metadata |
| `hrv_windows_normalized.csv` | Z-score normalized HRV | Same | Raw + z-norm versions of HRV features |
| `hrv_windows_pseudo_labeled.csv` | Clustered drowsiness labels | Same | Prior + `cluster_id`, `pseudo_kss`, `pseudo_label` |
| `hrv_windows_final.csv` | Smoothed labels, ready for ML | Same | Prior + `pseudo_kss_smoothed`, `pseudo_label_smoothed` |

---

## Configuration Reference

All configuration is centralized in [config.py](../src/config.py). Key settings:

### Dataset Paths
```python
DROZY_ROOT = PROJECT_ROOT / "data" / "raw" / "DROZY"
DDD_DIR = PROJECT_ROOT / "data" / "raw" / "DDD"
PSG_DIR = DROZY_ROOT / "psg"
KSS_FILE = DROZY_ROOT / "KSS.txt"
```

### Output Directories
```python
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ECG_CSV_DIR = PROJECT_ROOT / "data" / "preprocessed" / "ecg_csv"
PSEUDO_KSS_DIR = OUTPUTS_DIR / "pseudo_kss"
```

### ECG Parameters
```python
ECG_SAMPLE_RATE = 512  # Hz
FILTER_LOWCUT = 0.5    # Hz
FILTER_HIGHCUT = 40.0  # Hz
FILTER_ORDER = 4
RPEAK_METHOD = "pantompkins1985"
```

### RR-Interval Artifact Rejection
```python
RR_MIN_MS = 300
RR_MAX_MS = 2000
RR_OUTLIER_THRESHOLD = 2  # Standard deviations
MIN_RR_INTERVALS = 10
```

### HRV Windowing
```python
WINDOW_SEC = 60        # Window length
STEP_SEC = 20          # Stride between windows
```

### Feature Columns
```python
HRV_FEATURE_COLS = [
    "mean_rr", "mean_hr", "sdnn", "rmssd",
    "nn50", "pnn50", "cv_rr"
]
```

---

## Dependencies

The project uses the following Python packages:

### Core Scientific Stack
- `numpy >= 1.24`: Numerical computing
- `pandas >= 2.0`: Data manipulation and analysis
- `scipy >= 1.11`: Scientific computing utilities
- `matplotlib >= 3.7`: Visualization
- `seaborn >= 0.13`: Statistical visualization

### ECG & Signal Processing
- `pyedflib >= 0.1.28`: EDF file reading
- `neurokit2 >= 0.2.7`: Physiological signal processing (R-peak detection, HRV)

### Machine Learning
- `scikit-learn >= 1.3`: Classical ML, metrics, preprocessing
- `xgboost >= 1.7`: Gradient boosting
- `catboost >= 1.2`: Gradient boosting alternative

### Utilities
- `python-dotenv >= 1.0`: Environment variable loading
- `tqdm >= 4.66`: Progress bars

---

## Usage Instructions

### Prerequisites
1. **Set up virtual environment** (Python 3.12 recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or: source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure data paths** (create `.env` in project root):
   ```env
   DROZY_ROOT=data/raw/DROZY
   DDD_DIR=data/raw/DDD
   OUTPUTS_DIR=outputs
   ```

### Running the Pipeline

**Option 1: Full Pipeline (Step-by-Step)**
```bash
# Extract ECG from EDF files
python src/extract_ecg.py --dataset all

# Extract windowed HRV features
python src/hrv_extractor.py --dataset all

# Normalize HRV features per subject
python src/normalize_features.py

# Generate pseudo-labels via clustering
python src/generate_pseudo_kss.py

# Smooth temporal labels
python src/smooth_labels.py

# Final dataset ready for ML
# Output: outputs/hrv_windows_final.csv
```

**Option 2: Individual Steps**
```bash
# Extract only DROZY dataset
python src/extract_ecg.py --dataset drozy

# Normalize with subject exclusion
python src/normalize_features.py --exclude-subjects drozy_s04

# Custom clustering with 4 clusters
python src/generate_pseudo_kss.py --n-clusters 4

# Apply stricter smoothing
python src/smooth_labels.py --kernel-size 7 --enforce-monotonic
```

### Output Inspection
After running the pipeline, inspect outputs:
```bash
# View final data
head -20 outputs/hrv_windows_final.csv

# Check class distribution
pandas -c "outputs/hrv_windows_final.csv" \
       "df['pseudo_label_smoothed'].value_counts()"

# View cluster visualizations
# (Open PNG files in outputs/pseudo_kss/)
```

---

## Current Implementation Status

### ✅ Completed
- [x] Configuration management (config.py)
- [x] ECG extraction from EDF files (extract_ecg.py)
- [x] R-peak detection and RR-interval artifact rejection (hrv_metrics.py)
- [x] Windowed HRV feature extraction (hrv_extractor.py)
- [x] Per-subject feature normalization (normalize_features.py)
- [x] Pseudo-label generation via clustering (generate_pseudo_kss.py)
- [x] Temporal label smoothing (smooth_labels.py)
- [x] Subject-level train/val/test split utilities (split_utils.py)
- [x] End-to-end data pipeline for supervised learning

### 🔄 In Progress / Future Work
- [ ] ML model training (logistic regression, random forest, XGBoost)
- [ ] Deep learning models (LSTM, Transformer on temporal HRV sequences)
- [ ] Hyperparameter tuning and cross-validation
- [ ] Model evaluation metrics and ablation studies
- [ ] Inference pipeline for real-time drowsiness detection
- [ ] Webcam/video-based integration for practical deployment
- [ ] Additional physiological signals (e.g., eye gaze, facial landmarks)

---

## Key Design Decisions

### 1. **Windowed HRV over Single Values**
- Sliding windows with overlap capture temporal dynamics
- More training samples from same data
- Smoother temporal coverage

### 2. **Per-Subject Normalization**
- Removes inter-subject baseline physiological differences
- Enables clustering to detect **within-subject** drowsiness changes
- Improves generalization across diverse populations

### 3. **Unsupervised Pseudo-Labels**
- Leverages domain knowledge (HRV ↔ drowsiness)
- Enables training without hand-labeled data
- Validated against real labels when available (DROZY)

### 4. **Subject-Level Splits**
- Prevents data leakage (same subject across train/val/test)
- Ensures realistic evaluation of generalization
- Critical for practical deployment

### 5. **Temporal Label Smoothing**
- Reduces instantaneous HRV noise
- Enforces physiological plausibility
- Optional monotonic constraint for progressive drowsiness paradigms

---

## Troubleshooting

### Common Issues

**Issue: "EDF file not found"**
- Ensure dataset paths in `.env` are correct
- Check that DROZY and DDD folders contain EDF files

**Issue: "Insufficient RR intervals in window"**
- Increase `WINDOW_SEC` in config.py
- Reduce `STEP_SEC` to get shorter windows

**Issue: "Subject std == 0" warnings**
- Some subjects may have constant or near-constant signals
- Check data quality; consider exclusion if widespread

**Issue: Cluster validation (kappa, accuracy) low**
- Pseudo-labels may not align with real labels
- Consider adjusting number of clusters or normalization method
- Inspect raw HRV distributions by class

---

## References

1. **PhysDrive Dataset**: Multimodal remote physiological measurement dataset for driver monitoring
   - https://arxiv.org/html/2507.19172v1

2. **ECG-Based Driving Fatigue Detection**: Using Heart Rate Variability Analysis with Mutual Information
   - https://www.mdpi.com/2078-2489/14/10/539

3. **NeuroKit2**: Physiological signal processing in Python
   - https://neurokit2.readthedocs.io/

4. **Heart Rate Variability (HRV)**: Time-domain HRV metrics and their interpretation
   - Task Force of the European Society of Cardiology and NASPE

---

## Contact & Support

For questions or contributions, see:
- [README.md](../README.md) — Project overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Contribution guidelines
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) — Community standards

---

**Last Updated:** May 2026
**Project Status:** Data pipeline complete; ready for model training phase
