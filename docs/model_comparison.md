# Comparing the two models with the optimal threshold, we obtain the following metrics:

## Metrics Comparison After Optimal Thresholding (FPR=0.05):

### XGBoost:

Threshold 0.5336:
    - F1=0.9521
    - Prec=0.9337
    - Rec=0.9713
    - Confusion Matrix:
[[292 12]
[ 5 169]]

### CatBoost:

Threshold 0.2513:
    - F1=0.9556
    - Prec=0.9247
    - Rec=0.9885
    - Confusion Matrix:
[[290 14]
[ 2 172]]