# Final Project Report: ECG-Based Drowsiness Detection

**Project:** Sentinoids — Drowsiness Detection via HRV Analysis  
**Date:** May 2026  
**Branch:** issue-61/final-report

---

## Executive Summary

Driver drowsiness is a leading contributor to road traffic fatalities worldwide. This project developed an ECG-based drowsiness detection system using Heart Rate Variability (HRV) features extracted from two public physiological datasets: DROZY (14 subjects, 512 Hz) and DDD (10 subjects, 128 Hz). Because neither dataset provides densely annotated, per-window drowsiness labels suitable for supervised learning, a pseudo-labeling strategy was designed: K-Means clustering on per-subject z-normalized HRV features produces three classes (alert, transitional, drowsy), validated qualitatively against DROZY's session-level Karolinska Sleepiness Scale (KSS) ground truth.

An XGBoost classifier trained on four selected z-normalized HRV features — `sdnn_znorm`, `cv_rr_znorm`, `rmssd_znorm`, `pnn50_znorm` — achieves strong discriminative performance under Leave-One-Subject-Out cross-validation (LOSO-CV): PR-AUC 0.958, ROC-AUC 0.979, and a mean per-fold F1 of 0.88. These results demonstrate that time-domain HRV features carry a meaningful signal for distinguishing alert from drowsy physiological states at the subject-independent level.

However, the system's practical utility is tempered by significant limitations in label quality and dataset suitability. The pseudo-labels attain only moderate agreement with DROZY ground truth (Cohen's κ = 0.15, accuracy = 0.59), and stability analysis reveals that prediction sequences are frequently erratic — 17 of 56 DROZY recordings exceed a 30% label-flip rate, and only 3 of 20 DDD recordings display the expected monotonic fatigue buildup. These findings highlight the gap between classifying HRV states and reliably tracking drowsiness onset over time, and motivate concrete directions for future dataset collection and feature engineering.

---

## 1. Introduction

Drowsy driving is responsible for an estimated 6–21% of all road crashes, with impairment comparable to alcohol intoxication at severe fatigue levels. Detection systems that can alert a driver before microsleep occurs have clear life-safety value. Existing approaches span camera-based eye-tracking, steering-behavior analysis, and physiological sensing; each has distinct strengths and failure modes.

This project focuses on **ECG-based physiological sensing**, specifically **Heart Rate Variability** analysis. HRV captures the autonomic nervous system's regulation of cardiac rhythm: as cognitive load decreases and parasympathetic activity rises during drowsiness onset, characteristic changes occur in inter-beat interval statistics (lower SDNN, lower RMSSD, reduced pNN50). These changes are measurable with a simple single-lead ECG and are relatively robust to lighting conditions and facial occlusion — limitations that affect camera-based systems.

The project's primary objectives were:

1. Build a complete data pipeline from raw EDF ECG recordings to windowed, normalized, pseudo-labeled features.
2. Train and evaluate a subject-independent classifier using LOSO-CV to simulate real-world deployment.
3. Analyze temporal prediction stability, since deployment requires consistent, trustworthy alerts rather than isolated correct predictions.

---

## 2. Datasets

### 2.1 DROZY Dataset

The DROZY dataset (Massoz et al., 2016) contains polysomnographic recordings from **14 subjects** sampled at **512 Hz**. Each subject participated in multiple sessions designed to induce different levels of sleepiness. Crucially, DROZY provides **session-level KSS ratings** (Karolinska Sleepiness Scale, 1–9) collected once per recording, making it the project's only source of ground-truth drowsiness labels. Sessions are approximately **10 minutes** in duration.

After applying exclusion rules — sessions with KSS = 0 (test not performed), and recordings lost to backup failure — six session/trial combinations were removed: (7,1), (9,1), (10,2), (12,2), (12,3), and (13,3). The final DROZY contribution to the training dataset is **213 HRV windows** from 56 recordings.

**Key limitation:** A single KSS rating per session cannot capture within-session drowsiness dynamics. The label effectively describes the average state, not the progression from alert to drowsy within the recording.

### 2.2 DDD Dataset

The Driver Drowsiness Detection (DDD) dataset (Orosco et al., 2023) contains ECG recordings from **10 subjects** sampled at **128 Hz** during approximately **2-hour driving sessions**. Drowsiness annotations were collected via **button press by subjects** when they felt drowsy, making it a self-report measure rather than an external rating. The extended session length makes DDD well-suited for studying temporal fatigue buildup.

DDD contributes **2,328 HRV windows** — the dominant portion of the training set — across 20 recordings.

**Key limitation:** Button-press self-report is subjective and prone to reporting lag; subjects may press late or inconsistently. There is no expert-validated ground truth.

### 2.3 Combined Dataset Summary

| Property | DROZY | DDD | Combined |
|---|---|---|---|
| Subjects | 14 | 10 | 24 |
| Sample rate | 512 Hz | 128 Hz | — |
| Session length | ~10 min | ~2 hours | — |
| Annotation method | Expert KSS (session-level) | Self-report button press | — |
| HRV windows | 213 | 2,328 | **2,541** |
| Recordings | 56 | 20 | 76 |

---

## 3. Methodology

### 3.1 ECG Extraction and Preprocessing

Raw ECG signals were extracted from EDF (European Data Format) files using `pyedflib`. The extraction module performs flexible channel detection — matching on "ECG" or "EKG" labels, case-insensitively — and writes per-recording CSV files with columns `timestamp_sec` and `ecg`.

Prior to HRV computation, each ECG signal was bandpass filtered (**0.5–40 Hz, 4th-order Butterworth**) to suppress baseline wander (low-frequency drift from respiration and body movement) and high-frequency noise (myoelectric interference). R-peak detection was performed using the **Pan-Tompkins algorithm** (1985) as implemented in NeuroKit2.

RR intervals derived from detected peaks were subjected to artifact rejection:

- Physiological bounds: **300–2000 ms** (equivalent to 30–200 bpm)
- Ectopic beat removal: intervals deviating more than **20% from the local median** were discarded
- Windows with fewer than **50 clean RR intervals** were excluded entirely

### 3.2 Windowed HRV Feature Extraction

HRV features were computed over **5-minute sliding windows with a 1-minute step** (80% overlap). This window length is consistent with the Task Force of the European Society of Cardiology guidelines for short-term HRV analysis, which recommend at least 5 minutes for stable time-domain metrics.

Seven time-domain features were extracted per window:

| Feature | Description |
|---|---|
| `mean_rr` | Mean RR interval (ms) |
| `mean_hr` | Mean heart rate (bpm) |
| `sdnn` | Standard deviation of RR intervals (ms) |
| `rmssd` | Root mean square of successive RR differences (ms) |
| `nn50` | Count of successive RR differences > 50 ms |
| `pnn50` | Percentage of successive RR differences > 50 ms |
| `cv_rr` | Coefficient of variation (std / mean × 100%) |

### 3.3 Per-Subject Z-Score Normalization

Raw HRV features vary substantially across individuals due to differences in baseline physiology (age, fitness, resting heart rate). A subject with a naturally low SDNN of 30 ms cannot be compared directly to one with a resting SDNN of 70 ms. To isolate **within-subject** changes attributable to fatigue, each feature was z-score normalized per subject:

```
z_norm = (x − subject_mean) / subject_std
```

This produces z-normalized variants (`_znorm` suffix) that represent deviations from each person's personal baseline, enabling cross-subject clustering and classification. Subjects with constant signals (std = 0) were assigned `_znorm = 0` with a logged warning.

### 3.4 Feature Selection

Exploratory data analysis identified four z-normalized features as most informative for discriminating drowsiness states:

- `sdnn_znorm` — overall HRV; decreases with drowsiness
- `cv_rr_znorm` — relative variability; decreases with drowsiness
- `rmssd_znorm` — parasympathetic tone; decreases with drowsiness
- `pnn50_znorm` — proportion of large RR variations; decreases with drowsiness

These four features were used exclusively for clustering and model training. Raw (un-normalized) features were not used in the final model.

### 3.5 Pseudo-KSS Label Synthesis

Because neither dataset provides per-window drowsiness labels, a pseudo-labeling strategy was developed using domain knowledge about HRV changes during drowsiness.

**Clustering:** K-Means with **k = 3** clusters was applied to the four selected z-normalized features pooled across all subjects and datasets. Clusters were assigned drowsiness labels by ranking on mean `sdnn_znorm` (descending):

| Cluster rank | Mean SDNN | Pseudo-KSS | Class label |
|---|---|---|---|
| Highest | High HRV | 2 | Alert |
| Middle | Medium HRV | 5 | Transitional |
| Lowest | Low HRV | 8 | Drowsy |

The clustering produced a silhouette score of **0.25**, indicating modest but meaningful cluster separation — clusters are not well-isolated in the feature space, consistent with the gradual, continuous nature of drowsiness onset.

**Validation against DROZY ground truth:** Session-level KSS labels (1–9) were mapped to the same three classes (1–3: Alert, 4–6: Transitional, 7–9: Drowsy) and compared to pseudo-labels assigned to windows from those sessions. The pseudo-labels achieved:

- Classification accuracy: **0.59**
- Cohen's κ: **0.15** (slight agreement)

A κ of 0.15 is above chance but well below the threshold for reliable label agreement, confirming that the pseudo-labels are a noisy proxy for ground truth. This is an inherent limitation of the labeling approach and is discussed further in Section 7.

**Temporal smoothing:** A **median filter** (kernel size 3) was applied within each subject/trial to remove isolated label flips caused by transient HRV fluctuations. This produces the final `pseudo_kss_smoothed` and binary `pseudo_label_smoothed` columns. The binary threshold is **KSS ≥ 7** (drowsy).

### 3.6 Model Training

An **XGBoost** classifier was trained on the pseudo-labeled dataset using the four selected features. Class imbalance was addressed through sample weights proportional to inverse class frequency.

**Evaluation strategy:** Leave-One-Subject-Out cross-validation (LOSO-CV) was used as the primary evaluation protocol. In each fold, all windows from one subject are held out as the test set and the model is trained on the remaining 23 subjects. This simulates deployment on a new, unseen individual and prevents data leakage from within-subject temporal correlation.

A fixed subject-level split (approximately 80/10/10 for train/validation/test) was also maintained for hyperparameter tuning, with subjects assigned deterministically via the configuration's `SPLIT_MAP`.

---

## 4. Results

### 4.1 LOSO-CV Performance

| Metric | Value |
|---|---|
| PR-AUC | **0.958** |
| ROC-AUC | **0.979** |
| Mean per-fold F1 | **0.88** |

The high PR-AUC (0.958) is particularly meaningful given the class imbalance (36.84% overall drowsy rate), as PR curves are sensitive to false positives on imbalanced datasets. The ROC-AUC of 0.979 confirms strong overall discrimination between alert and drowsy windows across subjects.

These results should be interpreted with the caveat that the model is evaluated against **pseudo-labels**, not ground truth. A model can achieve high pseudo-label accuracy while having uncertain relationship to actual physiological drowsiness.

### 4.2 Feature Importance

Feature importance analysis identifies `sdnn_znorm` as the dominant predictor, consistent with the established literature linking overall HRV reduction to fatigue. The four features and their relative contributions:

| Feature | Relative Importance |
|---|---|
| `sdnn_znorm` | Highest |
| `cv_rr_znorm` | Second |
| `rmssd_znorm` | Third |
| `pnn50_znorm` | Lowest |

All four features contribute meaningfully, suggesting that the parasympathetic-specific metrics (`rmssd_znorm`, `pnn50_znorm`) add discriminative information beyond what `sdnn_znorm` alone captures.

### 4.3 Class Distribution

The overall pseudo-label drowsy rate across 2,541 windows is **36.84%**, indicating moderate class imbalance. The remaining windows are distributed between alert and transitional states.

---

## 5. Stability Analysis

Beyond single-window classification accuracy, practical deployment requires **temporal consistency**: predictions should not oscillate erratically between alert and drowsy across consecutive windows. Stability was assessed by computing the label-flip rate (fraction of consecutive window pairs with different predictions) per recording.

### 5.1 Overall Instability

Across all 56 DROZY recordings, **17 recordings (30.4%)** exceeded a 30% flip rate threshold, flagged as unstable. Unstable recordings tend to be DROZY sessions, which are only 10 minutes long — too short to accumulate a stable drowsiness trajectory.

### 5.2 DROZY-Specific Patterns

The 10-minute DROZY sessions are insufficiently long for the model to observe a meaningful temporal arc of alertness changing to drowsiness. With 5-minute windows and 1-minute steps, a 10-minute recording yields only 5–6 overlapping windows, which is a minimal sample for trend analysis. High flip rates in these recordings likely reflect genuine within-session HRV variability rather than meaningful state transitions.

### 5.3 DDD Temporal Patterns

The 2-hour DDD sessions provide enough windows to assess whether predictions follow the expected temporal pattern: predominantly alert at the start of a drive, with increasing drowsy predictions as the session progresses. Only **3 of 20 DDD recordings (15%)** display this expected monotonic fatigue buildup pattern.

The remaining 17 DDD recordings show flat, oscillating, or inverted patterns. Possible explanations include:

- Subjects may have started sessions already fatigued (no true alert baseline)
- Self-report button-press labels may not align well with underlying HRV-derived drowsiness
- Individual HRV responses to fatigue vary widely and may not follow group-level expectations

This finding raises questions about whether the pseudo-label synthesis is capturing the DDD subjects' actual fatigue trajectories.

### 5.4 Deployment Implications

For practical deployment, raw window-level predictions are not sufficient. A production system would require additional temporal post-processing — such as a rolling majority vote, exponential moving average, or a sequential model (HMM, LSTM) — before generating driver alerts. The high flip rates observed here suggest that predictions should not trigger alerts on individual windows.

---

## 6. Limitations

**1. Pseudo-label quality is low.** The fundamental constraint of this project is that neither dataset provides dense, per-window ground truth labels. The K-Means pseudo-labels achieve only Cohen's κ = 0.15 against DROZY session-level KSS — slight agreement by conventional standards. The XGBoost model therefore learns to predict a noisy approximation of drowsiness, not drowsiness itself. High LOSO-CV metrics reflect consistency of pseudo-label prediction, not accuracy against human-rated ground truth.

**2. No periodic KSS ground truth in either dataset.** DROZY provides one KSS rating per 10-minute session; DDD relies on self-report. Neither provides the dense, regularly-sampled expert ratings (e.g., every 4 minutes) needed to directly supervise per-window classifiers.

**3. DROZY sessions are too short for temporal analysis.** Ten-minute sessions yield 5–6 HRV windows per recording, limiting both temporal modeling and stability assessment. The high flip rates in DROZY recordings are partly an artifact of this brevity.

**4. Only time-domain HRV features.** The feature set is restricted to seven time-domain metrics. Frequency-domain features (LF/HF power ratio, total spectral power) and non-linear features (sample entropy, detrended fluctuation analysis) are established drowsiness markers and were not included. Their absence may limit the model's ability to capture autonomic regulation dynamics that time-domain features miss.

**5. Small subject pool.** With 24 subjects total (14 DROZY + 10 DDD), the dataset is small by machine learning standards. Subject-level LOSO-CV with 24 folds is statistically meaningful but may not capture population-level diversity in HRV responses to fatigue.

**6. DDD temporal patterns mostly do not match expected fatigue buildup.** Only 15% of DDD recordings show the expected progressive drowsiness arc. This may indicate label quality issues, subject-level heterogeneity, or misalignment between HRV-derived clusters and self-reported fatigue — each of which undermines confidence in the system's ability to track fatigue buildup in real-world driving.

**7. Single-modal sensing.** ECG-based HRV is informative but susceptible to motion artifacts, electrode placement, and non-fatigue confounds (exercise, stress, caffeine). A practical system benefits from sensor fusion.

---

## 7. Future Work

**Acquire a dataset with periodic expert KSS ratings.** The UL-DD dataset, which collects KSS ratings from subjects every 4 minutes during extended driving sessions, would provide supervision appropriate for per-window classifiers. This would allow direct training without pseudo-labels and enable ground-truth validation of the temporal pattern analysis.

**Add frequency-domain HRV features.** The LF/HF power ratio is a sensitive marker of sympatho-vagal balance; its reduction has been repeatedly linked to drowsiness in driving studies. Adding LF, HF, and LF/HF features alongside time-domain metrics may improve both clustering separation and classification accuracy.

**Add non-linear HRV features.** Sample entropy (complexity of RR interval series) and detrended fluctuation analysis (long-range temporal correlations) capture dynamical properties of autonomic regulation that linear statistics miss. These features tend to decrease with drowsiness and may provide complementary discriminative information.

**Real-time deployment via wearable ECG.** The current pipeline processes pre-recorded EDF files offline. Adapting the inference pipeline to stream from a wearable single-lead ECG device (chest strap or smart garment) would enable real-time driver monitoring. The inference pipeline (`predict.py`, loaded XGBoost model + saved scaler) is architecturally ready; the streaming interface and alert logic require development.

**Temporal modeling.** Replace or supplement the XGBoost window-level classifier with a sequential model (LSTM, GRU, or Hidden Markov Model) that explicitly models state transitions over time. This would address the erratic flip rates observed in the stability analysis and align the system's output with the gradually evolving nature of drowsiness.

**Multi-modal fusion.** Combining ECG-based HRV with complementary signals — eye tracking (PERCLOS, blink rate), EEG frontal theta, or steering entropy — would improve robustness. Each modality captures different aspects of cognitive impairment, and fusion tends to reduce false-positive rates.

---

## 8. Conclusion

This project successfully built a complete end-to-end pipeline for ECG-based drowsiness detection: from raw EDF recordings through bandpass filtering, R-peak detection, windowed HRV extraction, per-subject normalization, K-Means pseudo-labeling, temporal smoothing, and XGBoost training with LOSO-CV evaluation. The classifier achieves strong discriminative performance on pseudo-labeled data (PR-AUC 0.958, ROC-AUC 0.979, mean F1 0.88), demonstrating that time-domain HRV features carry a meaningful signal for distinguishing physiological states associated with drowsiness.

At the same time, the project surfaces important honest limitations: pseudo-labels are a weak proxy for ground truth (κ = 0.15), temporal prediction stability is inconsistent across recordings, and dataset characteristics — short DROZY sessions, self-report DDD annotations — constrain what can be reliably concluded. The path toward a clinically or commercially viable system runs through acquiring denser expert-rated labels and adding frequency-domain and non-linear HRV features.

The pipeline, codebase, and documented findings provide a solid foundation for that next phase.

---

## References

1. Massoz, Q., Langohr, T., François, C., & Verly, J. G. (2016). **The ULg multimodality drowsiness database (called DROZY) and examples of use.** In *2016 IEEE Winter Conference on Applications of Computer Vision (WACV)* (pp. 1–7). IEEE.

2. Orosco, L., Correa, A. G., Diez, P., & Laciar, E. (2023). **Driver drowsiness detection based on physiological signals: A systematic review.** *Sensors*, 23(3), 1764. *(DDD dataset reference.)*

3. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. (1996). **Heart rate variability: Standards of measurement, physiological interpretation, and clinical use.** *Circulation*, 93(5), 1043–1065.

4. Pan, J., & Tompkins, W. J. (1985). **A real-time QRS detection algorithm.** *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236.

5. Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., … & Chen, S. H. A. (2021). **NeuroKit2: A Python toolbox for neurophysiological signal processing.** *Behavior Research Methods*, 53(4), 1689–1696.
