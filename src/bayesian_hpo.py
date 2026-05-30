"""
Bayesian hyperparameter optimization for XGBoost and CatBoost classifiers.

This module provides Bayesian search-based hyperparameter tuning using Optuna
for drowsiness detection models. Supports both fixed split and LOSO-CV evaluation.

Features:
    - Objective functions for XGBoost and CatBoost optimization
    - Configurable search spaces with reasonable defaults
    - Support for both fixed split and LOSO cross-validation
    - Integrated with the project's dataloader and metrics

Usage:
    >>> from bayesian_hpo import tune_xgboost, tune_catboost
    >>> from dataloader import SubjectWiseDataLoader
    >>>
    >>> loader = SubjectWiseDataLoader()
    >>> loader.prepare()
    >>>
    >>> # Tune XGBoost on fixed split
    >>> best_params_xgb = tune_xgboost(loader, n_trials=100)
    >>> # Tune CatBoost with LOSO-CV
    >>> best_params_cat = tune_catboost(loader, loso=True, n_trials=100)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from catboost import CatBoostClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from dataloader import SubjectWiseDataLoader

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_STATE: int = 42
RECALL_FLOOR: float = 0.3


class XGBoostObjective:
    """Objective function for Bayesian optimization of XGBoost hyperparameters."""

    def __init__(
        self,
        loader: SubjectWiseDataLoader,
        loso: bool = False,
    ):
        """
        Initialize XGBoost objective function.

        Args:
            loader: SubjectWiseDataLoader instance with data already loaded/split.
            loso: If True, use LOSO cross-validation. If False, use fixed split.
        """
        self.loader = loader
        self.loso = loso
        self.scale_pos_weight = None

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Evaluate XGBoost model with hyperparameters sampled by the trial.

        Args:
            trial: Optuna trial object for sampling hyperparameters.

        Returns:
            Mean F1 score across evaluation fold(s).
        """
        # Define search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5, log=False),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10, log=False),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10, log=False),
        }

        try:
            if self.loso:
                f1_scores = self._eval_loso(params)
            else:
                f1_scores = self._eval_fixed(params)

            mean_f1 = np.mean(f1_scores)
            log.info(f"Trial {trial.number}: F1={mean_f1:.4f} (std={np.std(f1_scores):.4f})")
            return mean_f1
        except Exception as e:
            log.warning(f"Trial {trial.number} failed: {e}")
            return 0.0

    def _eval_fixed(self, params: dict[str, Any]) -> list[float]:
        """Evaluate on fixed train/val/test split."""
        train_df = self.loader.splits["train"]
        val_df = self.loader.splits["val"]
        test_df = self.loader.splits["test"]

        X_train = train_df[self.loader.feature_cols].values
        y_train = train_df["pseudo_label_smoothed"].values
        X_val = val_df[self.loader.feature_cols].values
        y_val = val_df["pseudo_label_smoothed"].values
        X_test = test_df[self.loader.feature_cols].values

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Compute scale_pos_weight
        n_pos = int(y_train.sum())
        if n_pos == 0:
            return [0.0]
        scale_pos_weight = (len(y_train) - n_pos) / n_pos

        # Train
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, verbose=0)

        # Evaluate on val
        y_val_pred = model.predict(X_val)

        f1_val = f1_score(y_val, y_val_pred, zero_division=0)

        return [f1_val]

    def _eval_loso(self, params: dict[str, Any]) -> list[float]:
        """Evaluate using Leave-One-Subject-Out cross-validation."""
        folds = self.loader.get_loso_folds()
        f1_scores = []

        for fold_idx, fold in enumerate(folds):
            train_df = fold["train"]
            test_df = fold["test"]

            X_train = train_df[self.loader.feature_cols].values
            y_train = train_df["pseudo_label_smoothed"].values
            X_test = test_df[self.loader.feature_cols].values
            y_test = test_df["pseudo_label_smoothed"].values

            # Skip if test set has only one class
            if len(np.unique(y_test)) < 2:
                log.debug(f"Fold {fold_idx}: skipping (only one class in test set)")
                continue

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Compute scale_pos_weight
            n_pos = int(y_train.sum())
            if n_pos == 0:
                continue
            scale_pos_weight = (len(y_train) - n_pos) / n_pos

            # Train
            model = XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train, verbose=0)

            # Evaluate
            y_test_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            f1_scores.append(f1)

        return f1_scores if f1_scores else [0.0]


class CatBoostObjective:
    """Objective function for Bayesian optimization of CatBoost hyperparameters."""

    def __init__(
        self,
        loader: SubjectWiseDataLoader,
        loso: bool = False,
    ):
        """
        Initialize CatBoost objective function.

        Args:
            loader: SubjectWiseDataLoader instance with data already loaded/split.
            loso: If True, use LOSO cross-validation. If False, use fixed split.
        """
        self.loader = loader
        self.loso = loso

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Evaluate CatBoost model with hyperparameters sampled by the trial.

        Args:
            trial: Optuna trial object for sampling hyperparameters.

        Returns:
            Mean F1 score across evaluation fold(s).
        """
        # Define search space
        params = {
            "iterations": trial.suggest_int("iterations", 50, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        }

        try:
            if self.loso:
                f1_scores = self._eval_loso(params)
            else:
                f1_scores = self._eval_fixed(params)

            mean_f1 = np.mean(f1_scores)
            log.info(f"Trial {trial.number}: F1={mean_f1:.4f} (std={np.std(f1_scores):.4f})")
            return mean_f1
        except Exception as e:
            log.warning(f"Trial {trial.number} failed: {e}")
            return 0.0

    def _eval_fixed(self, params: dict[str, Any]) -> list[float]:
        """Evaluate on fixed train/val/test split."""
        train_df = self.loader.splits["train"]
        val_df = self.loader.splits["val"]
        test_df = self.loader.splits["test"]

        X_train = train_df[self.loader.feature_cols].values
        y_train = train_df["pseudo_label_smoothed"].values
        X_val = val_df[self.loader.feature_cols].values
        y_val = val_df["pseudo_label_smoothed"].values
        X_test = test_df[self.loader.feature_cols].values

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Train
        model = CatBoostClassifier(
            **params,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        # Evaluate on val
        y_val_pred = model.predict(X_val)

        f1_val = f1_score(y_val, y_val_pred, zero_division=0)

        return [f1_val]

    def _eval_loso(self, params: dict[str, Any]) -> list[float]:
        """Evaluate using Leave-One-Subject-Out cross-validation."""
        folds = self.loader.get_loso_folds()
        f1_scores = []

        for fold_idx, fold in enumerate(folds):
            train_df = fold["train"]
            test_df = fold["test"]

            X_train = train_df[self.loader.feature_cols].values
            y_train = train_df["pseudo_label_smoothed"].values
            X_test = test_df[self.loader.feature_cols].values
            y_test = test_df["pseudo_label_smoothed"].values

            # Skip if test set has only one class
            if len(np.unique(y_test)) < 2:
                log.debug(f"Fold {fold_idx}: skipping (only one class in test set)")
                continue

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train
            model = CatBoostClassifier(
                **params,
                auto_class_weights="Balanced",
                verbose=0,
                random_seed=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_test_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)
            f1_scores.append(f1)

        return f1_scores if f1_scores else [0.0]


# ── main tuning functions ─────────────────────────────────────────────────────


def tune_xgboost(
    loader: SubjectWiseDataLoader,
    n_trials: int = 100,
    loso: bool = False,
    timeout: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Tune XGBoost hyperparameters using Bayesian optimization (Optuna).

    Args:
        loader: SubjectWiseDataLoader instance with data already loaded/split.
        n_trials: Number of optimization trials (default 100).
        loso: If True, use LOSO cross-validation. If False, use fixed split
              (default False).
        timeout: Maximum optimization time in seconds (default None = no limit).
        verbose: If True, print detailed logs (default True).

    Returns:
        Dictionary with best hyperparameters found.

    Example:
        >>> from dataloader import SubjectWiseDataLoader
        >>> loader = SubjectWiseDataLoader()
        >>> loader.prepare()
        >>> best_params = tune_xgboost(loader, n_trials=100)
        >>> print(best_params)
    """
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = XGBoostObjective(loader, loso=loso)
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    log.info(
        f"Starting XGBoost hyperparameter tuning "
        f"({n_trials} trials, {'LOSO' if loso else 'fixed split'})..."
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
    )

    best_params = study.best_params
    best_f1 = study.best_value

    log.info(f"XGBoost tuning complete. Best F1={best_f1:.4f}")
    log.info(f"Best hyperparameters: {best_params}")

    return {"best_params": best_params, "best_value": best_f1}


def tune_catboost(
    loader: SubjectWiseDataLoader,
    n_trials: int = 100,
    loso: bool = False,
    timeout: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Tune CatBoost hyperparameters using Bayesian optimization (Optuna).

    Args:
        loader: SubjectWiseDataLoader instance with data already loaded/split.
        n_trials: Number of optimization trials (default 100).
        loso: If True, use LOSO cross-validation. If False, use fixed split
              (default False).
        timeout: Maximum optimization time in seconds (default None = no limit).
        verbose: If True, print detailed logs (default True).

    Returns:
        Dictionary with best hyperparameters found.

    Example:
        >>> from dataloader import SubjectWiseDataLoader
        >>> loader = SubjectWiseDataLoader()
        >>> loader.prepare()
        >>> best_params = tune_catboost(loader, n_trials=100, loso=True)
        >>> print(best_params)
    """
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = CatBoostObjective(loader, loso=loso)
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    log.info(
        f"Starting CatBoost hyperparameter tuning "
        f"({n_trials} trials, {'LOSO' if loso else 'fixed split'})..."
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
    )

    best_params = study.best_params
    best_f1 = study.best_value

    log.info(f"CatBoost tuning complete. Best F1={best_f1:.4f}")
    log.info(f"Best hyperparameters: {best_params}")

    return {"best_params": best_params, "best_value": best_f1}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    loader = SubjectWiseDataLoader()
    loader.prepare()

    print("\n" + "=" * 70)
    print("XGBoost Tuning (Fixed Split)")
    print("=" * 70)
    best_xgb = tune_xgboost(loader, n_trials=10)

    print("\n" + "=" * 70)
    print("CatBoost Tuning (LOSO-CV)")
    print("=" * 70)
    best_catboost = tune_catboost(loader, n_trials=10, loso=True)
