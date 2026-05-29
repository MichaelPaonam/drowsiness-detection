"""
Subject-wise dataloader with 80-10-10 train/val/test split and LOSO cross-validation.

This module loads HRV features and provides:
1. Fixed 80-10-10 subject-level splits (SubjectWiseDataLoader.split_subjects)
2. Leave-One-Subject-Out (LOSO) cross-validation (SubjectWiseDataLoader.get_loso_folds)

Both approaches ensure subject-level separation to prevent data leakage.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import HRV_WINDOWS_FINAL_CSV, SELECTED_FEATURES
except ImportError:
    from config import HRV_WINDOWS_FINAL_CSV, SELECTED_FEATURES

log = logging.getLogger(__name__)


class SubjectWiseDataLoader:
    """
    Loads HRV data and splits it into train/val/test sets on a subject level.
    
    This ensures no subject appears in multiple splits, preventing data leakage.
    The split is approximately 80% train, 10% val, 10% test by number of subjects.
    """
    
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "pseudo_label_smoothed",
        random_state: int = 42,
    ):
        """
        Initialize the dataloader.
        
        Args:
            csv_path: Path to the HRV features CSV. Defaults to HRV_WINDOWS_FINAL_CSV from config.
            feature_cols: List of feature column names. Defaults to SELECTED_FEATURES from config.
            target_col: Name of the target label column. Defaults to "pseudo_label_smoothed".
            random_state: Random seed for reproducibility of subject-level split.
        """
        self.csv_path = csv_path or HRV_WINDOWS_FINAL_CSV
        self.feature_cols = feature_cols or SELECTED_FEATURES
        self.target_col = target_col
        self.random_state = random_state
        
        self.df = None
        self.splits = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the HRV features CSV file.
        
        Returns:
            The loaded DataFrame.
        
        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            KeyError: If required columns are missing.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        log.info(f"Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Validate required columns
        required_cols = ["subject_id"] + self.feature_cols + [self.target_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        log.info(f"Found {self.df['subject_id'].nunique()} unique subjects")
        
        return self.df
    
    def split_subjects(self) -> dict[str, pd.DataFrame]:
        """
        Split data into train/val/test sets at the subject level.
        
        Uses sklearn's train_test_split twice to achieve approximately 80-10-10 split:
        - First split: 80% train + val, 20% test (which becomes 10% of total)
        - Second split: 80% train, 20% val (which becomes ~8% and ~2% of total)
        
        This produces approximately 80% train, 10% val, 10% test by subject count.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing DataFrames for each split.
        
        Raises:
            ValueError: If df is not loaded yet.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get unique subjects
        unique_subjects = self.df["subject_id"].unique()
        n_subjects = len(unique_subjects)
        log.info(f"Splitting {n_subjects} subjects into train/val/test...")
        
        # First split: 80% train+val, 20% test
        train_val_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=0.1,  # 10% test
            random_state=self.random_state,
        )
        
        # Second split: 80% train, 20% val (from train+val)
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=1 / 9,  # 10% of 90% ≈ 10% overall for val
            random_state=self.random_state,
        )
        
        # Create split DataFrames
        self.splits["train"] = self.df[self.df["subject_id"].isin(train_subjects)].copy()
        self.splits["val"] = self.df[self.df["subject_id"].isin(val_subjects)].copy()
        self.splits["test"] = self.df[self.df["subject_id"].isin(test_subjects)].copy()
        
        # Log split statistics
        total_subjects = n_subjects
        for split_name, split_df in self.splits.items():
            n_rows = len(split_df)
            n_subj = split_df["subject_id"].nunique()
            label_dist = split_df[self.target_col].value_counts().to_dict()
            pct_rows = 100 * n_rows / len(self.df)
            pct_subj = 100 * n_subj / total_subjects
            log.info(
                f"{split_name}: {n_rows} rows ({pct_rows:.1f}%), {n_subj} subjects ({pct_subj:.1f}%), "
                f"labels: {label_dist}"
            )
        
        return self.splits
    
    def get_split(
        self,
        split_name: str,
        return_arrays: bool = False,
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
        """
        Get a specific split (train/val/test).
        
        Args:
            split_name: One of 'train', 'val', or 'test'.
            return_arrays: If True, return (X, y) as numpy arrays. If False, return DataFrame.
        
        Returns:
            If return_arrays=False: DataFrame with features and target.
            If return_arrays=True: Tuple of (X, y) numpy arrays.
        
        Raises:
            ValueError: If split_name is invalid or splits not created yet.
        """
        if split_name not in self.splits:
            raise ValueError(
                f"Split '{split_name}' not found. Available: {list(self.splits.keys())}"
            )
        
        df = self.splits[split_name]
        
        if return_arrays:
            X = df[self.feature_cols].values.astype(np.float32)
            y = df[self.target_col].values.astype(np.int64)
            return X, y
        
        return df
    
    def get_all_splits(
        self,
        return_arrays: bool = False,
    ) -> Union[dict[str, pd.DataFrame], dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Get all splits (train/val/test).
        
        Args:
            return_arrays: If True, return (X, y) tuples. If False, return DataFrames.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing either DataFrames
            or tuples of (X, y) numpy arrays.
        """
        result = {}
        for split_name in ["train", "val", "test"]:
            result[split_name] = self.get_split(split_name, return_arrays=return_arrays)
        return result
    
    def prepare(self) -> dict[str, pd.DataFrame]:
        """
        Load data and create splits (convenience method).
        
        Returns:
            Dictionary with train/val/test splits.
        """
        self.load_data()
        return self.split_subjects()
    
    def get_loso_folds(self) -> list[dict[str, pd.DataFrame]]:
        """
        Generate Leave-One-Subject-Out (LOSO) cross-validation folds.
        
        Each fold has one subject as test set and all others as train set.
        Returns list of N folds, where N = number of subjects.
        
        Returns:
            List of dictionaries, each with keys 'train' and 'test' containing DataFrames.
        
        Raises:
            ValueError: If data is not loaded yet.
        
        Example:
            >>> loader = SubjectWiseDataLoader()
            >>> loader.load_data()
            >>> folds = loader.get_loso_folds()
            >>> for fold_idx, fold in enumerate(folds):
            ...     train_df = fold['train']
            ...     test_df = fold['test']
            ...     # Train model on train_df, evaluate on test_df
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        unique_subjects = sorted(self.df["subject_id"].unique())
        folds = []
        
        log.info(f"Creating LOSO folds with {len(unique_subjects)} subjects...")
        
        for fold_idx, test_subject in enumerate(unique_subjects):
            # Test set: current subject
            test_df = self.df[self.df["subject_id"] == test_subject].copy()
            
            # Train set: all other subjects
            train_df = self.df[self.df["subject_id"] != test_subject].copy()
            
            fold = {
                "train": train_df,
                "test": test_df,
            }
            folds.append(fold)
            
            log.info(
                f"Fold {fold_idx+1:2d}/{len(unique_subjects)}: "
                f"test_subject={test_subject}, "
                f"train_samples={len(train_df)}, test_samples={len(test_df)}"
            )
        
        return folds
    
    def get_loso_fold(
        self,
        fold_idx: int,
        return_arrays: bool = False,
    ) -> Union[dict[str, pd.DataFrame], dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Get a specific LOSO fold by index.
        
        Args:
            fold_idx: Fold index (0 to num_subjects-1).
            return_arrays: If True, return (X, y) as numpy arrays. If False, return DataFrames.
        
        Returns:
            Dictionary with keys 'train' and 'test' containing either DataFrames
            or tuples of (X, y) numpy arrays.
        
        Raises:
            ValueError: If data not loaded or invalid fold index.
        """
        folds = self.get_loso_folds()
        
        if fold_idx < 0 or fold_idx >= len(folds):
            raise ValueError(
                f"Invalid fold_idx {fold_idx}. Must be in range [0, {len(folds)-1}]"
            )
        
        fold = folds[fold_idx]
        
        if return_arrays:
            return {
                "train": (
                    fold["train"][self.feature_cols].values.astype(np.float32),
                    fold["train"][self.target_col].values.astype(np.int64),
                ),
                "test": (
                    fold["test"][self.feature_cols].values.astype(np.float32),
                    fold["test"][self.target_col].values.astype(np.int64),
                ),
            }
        
        return fold


def load_dataset(
    csv_path: Optional[Path] = None,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "pseudo_label_smoothed",
    random_state: int = 42,
) -> Tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Convenience function to quickly load and split the dataset.
    
    Args:
        csv_path: Path to the HRV features CSV. Defaults to config.HRV_WINDOWS_FINAL_CSV.
        feature_cols: Feature column names. Defaults to config.SELECTED_FEATURES.
        target_col: Target label column name.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (splits_dict, feature_cols) where:
            - splits_dict: {'train': df, 'val': df, 'test': df}
            - feature_cols: The list of feature column names used
    """
    loader = SubjectWiseDataLoader(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_col=target_col,
        random_state=random_state,
    )
    splits = loader.prepare()
    return splits, loader.feature_cols


def load_loso_folds(
    csv_path: Optional[Path] = None,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "pseudo_label_smoothed",
) -> Tuple[list[dict[str, pd.DataFrame]], list[str]]:
    """
    Convenience function to load LOSO cross-validation folds.
    
    Args:
        csv_path: Path to the HRV features CSV. Defaults to config.HRV_WINDOWS_FINAL_CSV.
        feature_cols: Feature column names. Defaults to config.SELECTED_FEATURES.
        target_col: Target label column name.
    
    Returns:
        Tuple of (folds, feature_cols) where:
            - folds: List of N folds, each with {'train': df, 'test': df}
            - feature_cols: The list of feature column names used
    
    Example:
        >>> folds, features = load_loso_folds()
        >>> for fold_idx, fold in enumerate(folds):
        ...     X_train, y_train = fold['train'][features].values, fold['train']['pseudo_label_smoothed'].values
        ...     X_test, y_test = fold['test'][features].values, fold['test']['pseudo_label_smoothed'].values
    """
    loader = SubjectWiseDataLoader(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_col=target_col,
    )
    loader.load_data()
    folds = loader.get_loso_folds()
    return folds, loader.feature_cols


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Fixed 80-10-10 Split")
    print("="*70)
    
    # Create dataloader
    loader = SubjectWiseDataLoader()
    
    # Load and split
    splits = loader.prepare()
    
    # Get as numpy arrays for training
    train_data = loader.get_split("train", return_arrays=True)
    val_data = loader.get_split("val", return_arrays=True)
    test_data = loader.get_split("test", return_arrays=True)
    
    print("\nFeature columns used:", loader.feature_cols)
    print("Train shape:", train_data[0].shape, train_data[1].shape)
    print("Val shape:", val_data[0].shape, val_data[1].shape)
    print("Test shape:", test_data[0].shape, test_data[1].shape)
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Leave-One-Subject-Out (LOSO) Cross-Validation")
    print("="*70)
    
    # Get LOSO folds
    folds = loader.get_loso_folds()
    print(f"\nTotal LOSO folds: {len(folds)}")
    
    # Example: first 3 folds
    for fold_idx in range(min(3, len(folds))):
        fold = folds[fold_idx]
        X_train, y_train = fold["train"][loader.feature_cols].values, fold["train"]["pseudo_label_smoothed"].values
        X_test, y_test = fold["test"][loader.feature_cols].values, fold["test"]["pseudo_label_smoothed"].values
        
        print(f"Fold {fold_idx}: train {X_train.shape}, test {X_test.shape}")

