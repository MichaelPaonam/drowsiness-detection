# Drowsiness Detection

Drowsiness Detection is a data science project focused on identifying signs of fatigue from PPG signals, with an initial emphasis on ECG data. The project is intended to support early detection of drowsiness using HRV features generated from ECG data or BVP curves.

## Project Objective

The main goal is to build a system that classifies whether a person is:

- `alert`
- `drowsy`

## Approach

- Feature-based machine learning using HRV features extracted from ECG data and BVP curves.
- Time domain HRV features like sdnn and rmssd can have noticeable correlation with fatigue. Hence, models can be trained on them.

The recommended path is to begin with a feature-engineered baseline on various models and determine the best one.

## Planned Workflow

1. Define the task, label scheme, and evaluation metrics.
2. Collect or select suitable public datasets for drowsiness detection.
3. Build preprocessing pipelines to format and clean ECG data. Generate HRV features.
4. Perform exploratory data analysis to assess quality, balance, and edge cases.
5. Train baseline models and compare.
6. Evaluate performance with a strong focus on precision for drowsy cases to minimize false warnings.
7. Package the best model.

## Evaluation Priorities

This project should prioritize:

- High precision for drowsy events
- Low false-positive rate
- Near real-time inference performance

## Suggested Repository Structure

```text
drowsiness-detection/
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── docs
│   ├── boxplots.png
│   ├── FINAL_REPORT.md
│   ├── IMPLEMENTATION.md
│   └── model_comparison.md
├── LICENSE
├── notebooks
│   ├── correlation.ipynb
│   ├── feature_analysis.ipynb
│   ├── feature_selection.ipynb
│   └── train.ipynb
├── plan.md
├── pyproject.toml
├── pyrightconfig.json
├── README.md
├── requirements-dev.txt
├── requirements-optional.txt
├── requirements.txt
├── SECURITY.md
├── src
│   ├── bayesian_hpo.py
│   ├── config.py
│   ├── dataloader.py
│   ├── demo.py
│   ├── extract_ecg.py
│   ├── generate_pseudo_kss.py
│   ├── hrv_extractor.py
│   ├── hrv_metrics.py
│   ├── inference.py
│   ├── normalize_features.py
│   ├── smooth_labels.py
│   ├── split_utils.py
│   ├── stability_monitor.py
│   ├── threshold_utils.py
│   └── train_classifiers.py
└── tests
    ├── conftest.py
    ├── test_hrv_metrics.py
    ├── test_label_helpers.py
    ├── test_normalize_features.py
    └── test_split_utils.py
```

## Project Plan

The detailed implementation and milestone plan is documented in [plan.md](./plan.md).

## Tech Stack

- Python
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- PyTorch or TensorFlow
- MLflow or Weights & Biases

## Next Steps

- Remote Photoplethysmography (rPPG)

## References

- [ECG-Based Driving Fatigue Detection Using Heart Rate Variability Analysis with Mutual Information](https://www.mdpi.com/2078-2489/14/10/539)
- [Driving fatigue recognition model based on heart rate variability and respiratory rate](https://www.sciopen.com/article/10.16016/j.2097-0927.202203057)
- [PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring](https://arxiv.org/html/2507.19172v1#S3)

## Bots
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/MichaelPaonam/drowsiness-detection?utm_source=oss&utm_medium=github&utm_campaign=MichaelPaonam%2Fdrowsiness-detection&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
