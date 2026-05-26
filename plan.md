# Drowsiness Detection Project Plan

## 1. Project Goal

Build a data science system that detects human drowsiness from visual signals, with an initial focus on **camera-based driver monitoring**. The first version should classify whether a subject is **alert**, **possibly drowsy**, or **drowsy** using short video segments or frame sequences.

## 2. Problem Definition

### Primary objective
- Detect drowsiness early enough to support intervention before a dangerous event occurs.

### Target output
- Baseline: binary classification (`alert` vs `drowsy`)
- Extended: multi-class classification (`alert`, `fatigued`, `drowsy`)
- Optional: frame-level risk score between `0` and `1`

### Success criteria
- High recall for drowsy events
- Low false-negative rate
- Stable performance across lighting conditions, face angles, glasses, and different users
- Inference fast enough for near real-time use

## 3. Scope

### In scope
- Video/image-based drowsiness detection
- Eye closure, blink behavior, yawning, head pose, and facial landmarks
- Model training, evaluation, and reproducible experimentation

### Out of scope for phase 1
- EEG or wearable sensors
- Full production mobile app
- Regulatory certification

## 4. Key Questions

1. Which visual cues are most predictive of drowsiness in available datasets?
2. Does a feature-engineered approach outperform or complement deep learning on limited data?
3. How much temporal context is needed for reliable detection?
4. How robust is the model to low light, occlusion, and subject variation?

## 5. Data Strategy

### Candidate data sources
- Public drowsiness or fatigue detection datasets
- Driver monitoring video datasets
- Webcam-collected internal data for controlled experiments

### Data requirements
- Labeled sequences containing alert and drowsy states
- Diversity in age, gender presentation, skin tone, eyewear, and lighting
- Enough temporal coverage to capture blink rate, PERCLOS, yawns, and head nodding

### Labeling plan
- Define a label guide with clear rules for `alert`, `fatigued`, and `drowsy`
- Label at clip level first, then add frame/segment labels if needed
- Track uncertain samples separately rather than forcing noisy labels

### Data governance
- Document dataset licenses and usage limits
- Remove personally sensitive metadata where possible
- Store raw and processed data separately

## 6. Project Workflow

### Phase 1: Research and setup
- Review recent drowsiness detection approaches
- Identify 1 to 3 datasets suitable for benchmarking
- Define project metrics, label taxonomy, and experiment tracking approach

### Phase 2: Data ingestion and preprocessing
- Build loaders for images/videos and labels
- Extract frames or clips at consistent sampling rates
- Detect faces and facial landmarks
- Standardize crop size, frame rate, and sequence length
- Create train, validation, and test splits by subject to avoid leakage

### Phase 3: Exploratory data analysis
- Measure class balance and subject distribution
- Inspect lighting, pose, occlusion, and eyewear patterns
- Compare facial cue distributions across classes
- Identify noisy labels and weak samples

### Phase 4: Baseline modeling
- Baseline A: classical ML on engineered features
  - Features: eye aspect ratio, mouth aspect ratio, blink duration, PERCLOS, head pose
  - Models: logistic regression, random forest, XGBoost
- Baseline B: image CNN on single frames
- Baseline C: temporal model on frame sequences
  - Options: CNN + LSTM, 3D CNN, temporal transformer

### Phase 5: Evaluation and iteration
- Evaluate by subject-held-out test set
- Run ablations on feature groups and temporal window size
- Analyze failure cases by lighting, face angle, and accessories
- Calibrate thresholds for high-recall alerting

### Phase 6: Prototype deployment
- Package best model into an inference pipeline
- Add webcam or video-file demo
- Measure latency, CPU/GPU usage, and prediction stability

## 7. Modeling Approach

### Feature-engineered path
- Detect face and landmarks
- Compute eye closure and yawn metrics over time
- Aggregate temporal statistics over rolling windows
- Train interpretable classifiers

### Deep learning path
- Train on cropped face sequences
- Use augmentation for brightness, blur, rotation, and occlusion
- Compare frame-only vs temporal models

### Recommended initial strategy
- Start with engineered features plus a lightweight temporal classifier
- Use that baseline to validate labels and pipeline quality
- Move to deep temporal models once the data pipeline is reliable

## 8. Evaluation Plan

### Core metrics
- Recall for drowsy class
- Precision, F1-score
- ROC-AUC and PR-AUC
- Confusion matrix

### Operational metrics
- False negatives per hour
- Alert stability over time
- Average inference latency per frame or clip

### Validation design
- Subject-independent split
- Cross-validation when dataset size is limited
- External dataset test if available

## 9. Tools and Stack

### Recommended stack
- Python
- Jupyter for exploration
- pandas, numpy, matplotlib, seaborn
- OpenCV, MediaPipe or dlib for face/landmark processing
- scikit-learn for baselines
- PyTorch or TensorFlow for deep learning
- MLflow or Weights & Biases for experiment tracking

## 10. Deliverables

- Problem statement and metric definition
- Dataset inventory and labeling guide
- Reproducible preprocessing pipeline
- EDA report with data quality findings
- Baseline models and benchmark results
- Final trained prototype model
- Demo script for webcam or recorded video inference
- Final report summarizing results, limitations, and next steps

## 11. Risks and Mitigations

### Risk: weak or inconsistent labels
- Mitigation: define labeling rules early and audit samples manually

### Risk: data leakage across subjects
- Mitigation: split by identity, not by frame

### Risk: poor generalization in real-world lighting
- Mitigation: include augmentation and stress-test by condition

### Risk: temporal instability and noisy alerts
- Mitigation: smooth predictions across time and calibrate thresholds

## 12. Milestones

### Week 1
- Define scope, task, metrics, and datasets

### Week 2
- Build ingestion, preprocessing, and dataset split pipeline

### Week 3
- Complete EDA and train first classical baseline

### Week 4
- Train first deep learning baseline and compare results

### Week 5
- Run error analysis and improve robustness

### Week 6
- Build demo and finalize report

## 13. Suggested Repository Structure

```text
drowsiness-detection/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── reports/
├── experiments/
├── requirements.txt
└── plan.md
```

## 14. Immediate Next Steps

1. Select the primary use case and label definition.
2. Choose the first public dataset to benchmark on.
3. Implement the preprocessing pipeline with subject-level splits.
4. Build a feature-based baseline before training deep temporal models.
  