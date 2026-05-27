# Drowsiness Detection

Drowsiness Detection is a data science project focused on identifying signs of fatigue from visual signals, with an initial emphasis on camera-based driver monitoring. The project is intended to support early detection of drowsiness using facial and behavioral cues such as eye closure, blink patterns, yawning, and head pose.

## Project Objective

The main goal is to build a system that classifies whether a person is:

- `alert`
- `possibly drowsy`
- `drowsy`

The initial version may also start with a simpler binary task:

- `alert`
- `drowsy`

## Approach

The project will compare two main modeling strategies:

- Feature-based machine learning using signals like eye aspect ratio, mouth aspect ratio, blink duration, PERCLOS, and head pose
- Deep learning using image frames or short video sequences with temporal modeling

The recommended path is to begin with a feature-engineered baseline to validate the data pipeline and labels, then move to temporal deep learning models once the preprocessing workflow is stable.

## Planned Workflow

1. Define the task, label scheme, and evaluation metrics.
2. Collect or select suitable public datasets for drowsiness detection.
3. Build preprocessing pipelines for video loading, face detection, landmark extraction, and subject-level dataset splitting.
4. Perform exploratory data analysis to assess quality, balance, and edge cases.
5. Train baseline models and compare classical and deep learning approaches.
6. Evaluate performance with a strong focus on recall for drowsy cases.
7. Package the best model into a prototype inference pipeline for webcam or recorded video input.

## Evaluation Priorities

This project should prioritize:

- High recall for drowsy events
- Low false-negative rate
- Robustness across lighting, pose, eyewear, and user differences
- Near real-time inference performance

## Suggested Repository Structure

```text
drowsiness-detection/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
|-- notebooks/
|-- src/
|   |-- data/
|   |-- features/
|   |-- models/
|   |-- evaluation/
|   `-- inference/
|-- reports/
|-- experiments/
|-- README.md
`-- plan.md
```

## Project Plan

The detailed implementation and milestone plan is documented in [plan.md](./plan.md).

## Recommended Stack

- Python
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- OpenCV
- MediaPipe or dlib
- scikit-learn
- PyTorch or TensorFlow
- MLflow or Weights & Biases

## Next Steps

- Select the first dataset for benchmarking
- Define the exact label taxonomy
- Build the preprocessing and train/validation/test split pipeline
- Train a feature-based baseline model

## References

- [ECG-Based Driving Fatigue Detection Using Heart Rate Variability Analysis with Mutual Information](https://www.mdpi.com/2078-2489/14/10/539)
- [Driving fatigue recognition model based on heart rate variability and respiratory rate](https://www.sciopen.com/article/10.16016/j.2097-0927.202203057)
- [PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring](https://arxiv.org/html/2507.19172v1#S3)
