# -*- coding: utf-8 -*-
"""
Kepler Exoplanet Classification Project - Fixed Version

A comprehensive machine learning pipeline for detecting exoplanets
using Kepler time-series data with local CSV files.

Author: Data Science Team
Date: 2025-10-05
"""

import os
import warnings
from typing import Dict, Tuple, Optional, Any
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix)

# External libraries
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class Config:
    """Configuration class for hyperparameters and settings."""
    
    # Data files
    TRAIN_FILE = "exoTrain.csv"
    TEST_FILE = "exoTest.csv"
    
    # Data processing
    OUTLIER_THRESHOLD = 3  # IQR multiplier for outlier detection
    IMPUTATION_STRATEGY = 'median'
    
    # Feature engineering
    ROLLING_WINDOWS = [10, 50, 100]
    FFT_PERCENTILE = 85
    HISTOGRAM_BINS = 50
    
    # Feature selection
    TARGET_FEATURES = 200
    UNIVARIATE_K = 400  # 2x target for combined selection
    MUTUAL_INFO_K = 400
    IMPORTANCE_N = 400
    
    # Model training
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'is_unbalance': True,
        'seed': RANDOM_STATE
    }
    
    # Cross-validation
    CV_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50
    NUM_BOOST_ROUNDS = 500
    
    # SMOTE - Fixed configuration
    SMOTE_RATIO = 0.3  # Target ratio for minority class
    MIN_SAMPLES_FOR_SMOTE = 6  # Minimum samples needed to apply SMOTE
    SMOTE_K_NEIGHBORS = 3  # Reduced from default 5
    
    # Test split
    TEST_SIZE = 0.2


class DataLoader:
    """Class for loading and initial data validation."""
    
    @staticmethod
    def load_local_dataset(filepath: str) -> pd.DataFrame:
        """Load dataset from local CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        if 'LABEL' in df.columns:
            print(f"Label distribution:\n{df['LABEL'].value_counts().sort_index()}")
        
        return df
    
    @staticmethod
    def load_train_test_datasets() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load both training and test datasets."""
        print("=== LOADING DATASETS ===")
        
        # Load training data
        train_df = DataLoader.load_local_dataset(Config.TRAIN_FILE)
        
        # Try to load test data
        test_df = None
        try:
            test_df = DataLoader.load_local_dataset(Config.TEST_FILE)
            print("Both training and test datasets loaded successfully.")
        except FileNotFoundError:
            print(f"Test file '{Config.TEST_FILE}' not found. Will use train-test split.")
        
        return train_df, test_df


class DataPreprocessor:
    """Handles data cleaning and basic preprocessing."""
    
    def __init__(self):
        self.scaler: Optional[RobustScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and prepare the dataset."""
        print("Starting data cleaning...")
        
        # Create copy to avoid modifying original
        df_clean = df.copy()
        
        # Separate features and labels
        if 'LABEL' not in df_clean.columns:
            raise ValueError("Dataset must contain 'LABEL' column")
            
        X = df_clean.drop('LABEL', axis=1)
        y = df_clean['LABEL']
        
        # Handle missing values
        missing_count = X.isnull().sum().sum()
        print(f"Missing values found: {missing_count}")
        
        if missing_count > 0:
            self.imputer = SimpleImputer(strategy=Config.IMPUTATION_STRATEGY)
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = X
            
        # Handle outliers using IQR method
        X_clean = self._handle_outliers(X_imputed)
        
        print(f"Data cleaning completed. Final shape: {X_clean.shape}")
        return X_clean, y
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme outliers using IQR method."""
        print("Handling outliers...")
        
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for extreme outliers
        lower_bound = Q1 - Config.OUTLIER_THRESHOLD * IQR
        upper_bound = Q3 + Config.OUTLIER_THRESHOLD * IQR
        
        # Clip outliers (Winsorization)
        X_clean = X.clip(lower=lower_bound, upper=upper_bound, axis=1)
        
        outlier_count = ((X < lower_bound) | (X > upper_bound)).sum().sum()
        print(f"Outliers handled: {outlier_count}")
        
        return X_clean
    
    def scale_features(self, X_train: pd.DataFrame, 
                      X_test: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale features using RobustScaler."""
        print("Scaling features...")
        
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
            
        return X_train_scaled, None


class FeatureEngineer:
    """Advanced feature engineering for time-series astronomical data."""
    
    def create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive statistical features."""
        print("Creating statistical features...")
        
        features = pd.DataFrame(index=X.index)
        
        # Basic statistics
        features['mean_flux'] = X.mean(axis=1)
        features['median_flux'] = X.median(axis=1)
        features['std_flux'] = X.std(axis=1)
        features['var_flux'] = X.var(axis=1)
        features['min_flux'] = X.min(axis=1)
        features['max_flux'] = X.max(axis=1)
        features['range_flux'] = features['max_flux'] - features['min_flux']
        
        # Quantiles and percentiles
        features['q25_flux'] = X.quantile(0.25, axis=1)
        features['q75_flux'] = X.quantile(0.75, axis=1)
        features['iqr_flux'] = features['q75_flux'] - features['q25_flux']
        features['p10_flux'] = X.quantile(0.10, axis=1)
        features['p90_flux'] = X.quantile(0.90, axis=1)
        
        # Distribution shape measures
        features['skewness'] = X.skew(axis=1)
        features['kurtosis'] = X.kurtosis(axis=1)
        
        # Coefficient of variation
        features['cv_flux'] = features['std_flux'] / (features['mean_flux'] + 1e-8)
        
        return features
    
    def create_rolling_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistical features."""
        print("Creating rolling window features...")
        
        features = pd.DataFrame(index=X.index)
        
        for window in Config.ROLLING_WINDOWS:
            if window < X.shape[1]:
                rolling_data = pd.DataFrame(X.values).rolling(window=window, axis=1)
                
                features[f'rolling_mean_{window}'] = rolling_data.mean().iloc[:, -1]
                features[f'rolling_std_{window}'] = rolling_data.std().iloc[:, -1]
                features[f'rolling_max_{window}'] = rolling_data.max().iloc[:, -1]
                features[f'rolling_min_{window}'] = rolling_data.min().iloc[:, -1]
                
        return features
    
    def create_frequency_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create frequency domain features using FFT."""
        print("Creating frequency domain features...")
        
        features = pd.DataFrame(index=X.index)
        
        for i, row in X.iterrows():
            fft_values = np.abs(fft(row.values))
            
            # Spectral features
            freq_range = np.arange(len(fft_values))
            total_power = np.sum(fft_values)
            
            if total_power > 0:
                features.loc[i, 'spectral_centroid'] = np.sum(freq_range * fft_values) / total_power
                features.loc[i, 'spectral_rolloff'] = np.percentile(fft_values, Config.FFT_PERCENTILE)
                
                centroid = features.loc[i, 'spectral_centroid']
                features.loc[i, 'spectral_bandwidth'] = np.sqrt(
                    np.sum(((freq_range - centroid) ** 2) * fft_values) / total_power
                )
            else:
                features.loc[i, 'spectral_centroid'] = 0
                features.loc[i, 'spectral_rolloff'] = 0
                features.loc[i, 'spectral_bandwidth'] = 0
            
            # Spectral energy
            features.loc[i, 'spectral_energy'] = np.sum(fft_values ** 2)
            
            # Peak analysis
            peaks, _ = find_peaks(fft_values, height=np.mean(fft_values))
            features.loc[i, 'num_peaks'] = len(peaks)
            features.loc[i, 'peak_prominence'] = np.mean(fft_values[peaks]) if len(peaks) > 0 else 0
            
        return features
    
    def create_astronomical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific astronomical features."""
        print("Creating astronomical features...")
        
        features = pd.DataFrame(index=X.index)
        
        # Temporal variability
        diff_data = np.diff(X.values, axis=1)
        features['mean_diff'] = np.mean(diff_data, axis=1)
        features['std_diff'] = np.std(diff_data, axis=1)
        features['max_diff'] = np.max(np.abs(diff_data), axis=1)
        
        # Linear trend analysis
        for i, row in X.iterrows():
            x_vals = np.arange(len(row))
            try:
                slope, _, r_value, p_value, _ = stats.linregress(x_vals, row.values)
                features.loc[i, 'trend_slope'] = slope
                features.loc[i, 'trend_r_squared'] = r_value ** 2
                features.loc[i, 'trend_p_value'] = p_value
            except:
                features.loc[i, 'trend_slope'] = 0
                features.loc[i, 'trend_r_squared'] = 0
                features.loc[i, 'trend_p_value'] = 1
        
        # Autocorrelation features
        for lag in [1, 5, 10]:
            features[f'autocorr_lag_{lag}'] = X.apply(
                lambda row: row.autocorr(lag=lag) if not pd.isna(row.autocorr(lag=lag)) else 0, 
                axis=1
            )
        
        # Zero crossings (periodicity indicator)
        features['zero_crossings'] = X.apply(
            lambda row: np.sum(np.diff(np.sign(row - row.mean())) != 0), 
            axis=1
        )
        
        # Shannon entropy
        for i, row in X.iterrows():
            hist, _ = np.histogram(row.values, bins=Config.HISTOGRAM_BINS, density=True)
            hist = hist[hist > 0]  # Avoid log(0)
            features.loc[i, 'entropy'] = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
            
        return features
    
    def create_all_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature engineering methods."""
        print("Starting comprehensive feature engineering...")
        
        # Create all feature types
        stat_features = self.create_statistical_features(X)
        rolling_features = self.create_rolling_features(X)
        freq_features = self.create_frequency_features(X)
        astro_features = self.create_astronomical_features(X)
        
        # Combine all features
        all_features = pd.concat([
            X,  # Original features
            stat_features,
            rolling_features,
            freq_features,
            astro_features
        ], axis=1)
        
        print(f"Feature engineering completed. Total features: {all_features.shape[1]}")
        return all_features


class FeatureSelector:
    """Advanced feature selection using multiple methods."""
    
    def __init__(self):
        self.selected_features: Optional[list] = None
        self.feature_scores: Optional[pd.DataFrame] = None
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[np.ndarray, list, pd.DataFrame]:
        """Univariate feature selection using F-score."""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'selected': selected_mask
        }).sort_values('score', ascending=False)
        
        return X_selected, selected_features, feature_scores
    
    def mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
        """Feature selection using mutual information."""
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        
        top_k_indices = np.argsort(mi_scores)[-k:]
        selected_features = X.columns[top_k_indices].tolist()
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        return X.iloc[:, top_k_indices], selected_features, feature_scores
    
    def importance_based_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
        """Feature selection using Random Forest importance."""
        rf = RandomForestClassifier(
            n_estimators=200, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-n_features:]
        selected_features = X.columns[top_indices].tolist()
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return X.iloc[:, top_indices], selected_features, feature_scores
    
    def combined_selection(self, X: pd.DataFrame, y: pd.Series, final_k: int) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
        """Combine multiple feature selection methods."""
        print("Applying combined feature selection...")
        
        # Apply individual selection methods
        print("- Univariate selection...")
        _, uni_features, uni_scores = self.univariate_selection(X, y, k=Config.UNIVARIATE_K)
        
        print("- Mutual information selection...")
        _, mi_features, mi_scores = self.mutual_info_selection(X, y, k=Config.MUTUAL_INFO_K)
        
        print("- Importance-based selection...")
        _, imp_features, imp_scores = self.importance_based_selection(X, y, n_features=Config.IMPORTANCE_N)
        
        # Combine and normalize scores
        combined_scores = self._combine_scores(uni_scores, mi_scores, imp_scores)
        
        # Select top features
        top_features = combined_scores.nlargest(final_k, 'combined_score')['feature'].tolist()
        X_final = X[top_features]
        
        self.selected_features = top_features
        self.feature_scores = combined_scores
        
        print(f"Combined feature selection completed. Selected: {len(top_features)} features")
        return X_final, top_features, combined_scores
    
    def _combine_scores(self, uni_scores: pd.DataFrame, mi_scores: pd.DataFrame, 
                       imp_scores: pd.DataFrame) -> pd.DataFrame:
        """Combine and normalize feature scores from different methods."""
        # Merge scores
        combined = pd.merge(uni_scores[['feature', 'score']], 
                           mi_scores[['feature', 'mi_score']], on='feature')
        combined = pd.merge(combined, 
                           imp_scores[['feature', 'importance']], on='feature')
        
        # Normalize scores to [0, 1]
        for col in ['score', 'mi_score', 'importance']:
            col_min, col_max = combined[col].min(), combined[col].max()
            if col_max > col_min:
                combined[f'{col}_norm'] = (combined[col] - col_min) / (col_max - col_min)
            else:
                combined[f'{col}_norm'] = 0
        
        # Weighted combination
        combined['combined_score'] = (
            combined['score_norm'] * 0.4 +
            combined['mi_score_norm'] * 0.3 +
            combined['importance_norm'] * 0.3
        )
        
        return combined.sort_values('combined_score', ascending=False)
    
    def plot_feature_importance(self, top_n: int = 20):
        """Visualize top feature importance."""
        if self.feature_scores is not None:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_scores.head(top_n)
            
            sns.barplot(data=top_features, x='combined_score', y='feature')
            plt.title(f'Top {top_n} Most Important Features')
            plt.xlabel('Combined Score')
            plt.tight_layout()
            plt.show()


class SmartSampler:
    """Intelligent sampling strategy that handles small datasets robustly."""
    
    @staticmethod
    def get_safe_sampling_strategy(y_train: pd.Series, base_ratio: float = 0.3) -> Dict[str, Any]:
        """Determine safe sampling strategy based on class distribution."""
        class_counts = Counter(y_train)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]
        
        print(f"Class distribution: {dict(class_counts)}")
        
        # Calculate safe target count for minority class
        target_minority_count = min(
            int(majority_count * base_ratio),
            minority_count * 2  # Don't increase by more than 2x to be conservative
        )
        
        # Ensure we don't exceed original minority count if it's already balanced
        current_ratio = minority_count / majority_count
        if current_ratio > base_ratio:
            target_minority_count = minority_count
        
        # Calculate safe k_neighbors
        safe_k = min(Config.SMOTE_K_NEIGHBORS, minority_count - 1, 5)
        safe_k = max(1, safe_k)  # Ensure at least 1
        
        return {
            'strategy': {minority_class: target_minority_count},
            'can_apply': minority_count >= Config.MIN_SAMPLES_FOR_SMOTE,
            'k_neighbors': safe_k,
            'original_counts': class_counts,
            'target_minority_count': target_minority_count,
            'minority_count': minority_count
        }
    
    @staticmethod
    def apply_smart_sampling(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply intelligent sampling based on data characteristics."""
        sampling_config = SmartSampler.get_safe_sampling_strategy(y_train, Config.SMOTE_RATIO)
        
        if not sampling_config['can_apply']:
            print(f"⚠️  Skipping SMOTE: Not enough samples in minority class "
                  f"(need >= {Config.MIN_SAMPLES_FOR_SMOTE}, got {sampling_config['minority_count']})")
            return X_train, y_train
        
        # If target count is same as original, skip sampling
        if sampling_config['target_minority_count'] <= sampling_config['minority_count']:
            print("⚠️  Skipping SMOTE: Classes already balanced or target not higher than current")
            return X_train, y_train
        
        try:
            # Try SMOTE first
            sampler = SMOTE(
                sampling_strategy=sampling_config['strategy'],
                k_neighbors=sampling_config['k_neighbors'],
                random_state=RANDOM_STATE
            )
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print(f"✅ SMOTE applied successfully with k_neighbors={sampling_config['k_neighbors']}")
            
        except ValueError as e:
            print(f"⚠️  SMOTE failed: {str(e)}")
            
            # Try with even fewer neighbors
            if "n_neighbors" in str(e) and sampling_config['k_neighbors'] > 1:
                try:
                    reduced_k = max(1, sampling_config['k_neighbors'] - 1)
                    print(f"⚠️  Trying with k_neighbors={reduced_k}")
                    
                    sampler = SMOTE(
                        sampling_strategy=sampling_config['strategy'],
                        k_neighbors=reduced_k,
                        random_state=RANDOM_STATE
                    )
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                    print(f"✅ SMOTE applied with reduced k_neighbors={reduced_k}")
                    
                except ValueError as e2:
                    print(f"⚠️  SMOTE with reduced k failed: {str(e2)}")
                    
                    # Try BorderlineSMOTE as last resort
                    try:
                        print("⚠️  Trying BorderlineSMOTE as fallback")
                        sampler = BorderlineSMOTE(
                            sampling_strategy=sampling_config['strategy'],
                            k_neighbors=reduced_k,
                            random_state=RANDOM_STATE
                        )
                        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                        print(f"✅ BorderlineSMOTE applied successfully")
                        
                    except ValueError as e3:
                        print(f"❌ All sampling methods failed: {str(e3)}")
                        print("Using original data without oversampling")
                        return X_train, y_train
            else:
                print("❌ SMOTE failed and no fallback available. Using original data.")
                return X_train, y_train
        
        print(f"Sampling result: {Counter(y_train)} → {Counter(y_resampled)}")
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


class ModelTrainer:
    """LightGBM model trainer with cross-validation and robust sampling."""
    
    def __init__(self):
        self.model: Optional[lgb.Booster] = None
        self.cv_results: Dict[str, list] = {}
    
    def train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train model with stratified cross-validation and robust sampling."""
        print("Starting model training with cross-validation...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Train distribution: {Counter(y_train)}")
        print(f"Test distribution: {Counter(y_test)}")
        
        # Cross-validation
        cv_results = self._cross_validate(X_train, y_train)
        
        # Final model training
        final_model = self._train_final_model(X_train, y_train, X_test, y_test)
        
        return {
            'model': final_model,
            'cv_results': cv_results,
            'test_results': self._evaluate_model(final_model, X_test, y_test),
            'X_test': X_test,
            'y_test': y_test
        }
    
    def _cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, list]:
        """Perform stratified cross-validation with robust sampling."""
        skf = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        cv_results = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'confusion_matrices': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            print(f"\nTraining fold {fold}/{Config.CV_FOLDS}")
            
            # Split fold data
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            print(f"Fold {fold} - Train: {Counter(y_tr)}, Val: {Counter(y_val)}")
            
            # Apply smart sampling to training fold only
            X_tr_sampled, y_tr_sampled = SmartSampler.apply_smart_sampling(X_tr, y_tr)
            
            # Train model
            model = self._train_lgb_model(X_tr_sampled, y_tr_sampled, X_val, y_val)
            
            # Evaluate
            y_pred_val = (model.predict(X_val, num_iteration=model.best_iteration) >= 0.5).astype(int)
            
            # Store metrics
            cv_results['accuracy'].append(accuracy_score(y_val, y_pred_val))
            cv_results['precision'].append(precision_score(y_val, y_pred_val, zero_division=0))
            cv_results['recall'].append(recall_score(y_val, y_pred_val, zero_division=0))
            cv_results['f1'].append(f1_score(y_val, y_pred_val, zero_division=0))
            cv_results['confusion_matrices'].append(confusion_matrix(y_val, y_pred_val))
        
        # Print CV results
        self._print_cv_results(cv_results)
        return cv_results
    
    def _train_lgb_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> lgb.Booster:
        """Train LightGBM model."""
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(
            Config.LGBM_PARAMS,
            lgb_train,
            num_boost_round=Config.NUM_BOOST_ROUNDS,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'eval'],
            callbacks=[lgb.early_stopping(stopping_rounds=Config.EARLY_STOPPING_ROUNDS),
                      lgb.log_evaluation(0)]  # Suppress training logs
        )
        
        return model
    
    def _train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> lgb.Booster:
        """Train final model on full training set."""
        print("\nTraining final model...")
        
        # Apply smart sampling to full training set
        X_train_sampled, y_train_sampled = SmartSampler.apply_smart_sampling(X_train, y_train)
        
        # Train final model
        self.model = self._train_lgb_model(X_train_sampled, y_train_sampled, X_test, y_test)
        return self.model
    
    def _evaluate_model(self, model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model on test set."""
        y_pred_test = (model.predict(X_test, num_iteration=model.best_iteration) >= 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test
        }
    
    def _print_cv_results(self, cv_results: Dict[str, list]):
        """Print cross-validation results."""
        print("\n=== CROSS-VALIDATION RESULTS ===")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            scores = cv_results[metric]
            print(f"{metric.capitalize()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    def evaluate_with_thresholds(self, model: lgb.Booster, X_test: pd.DataFrame, 
                                y_test: pd.Series, thresholds: np.ndarray = None):
        """Evaluate model with different probability thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.1)
        
        y_prob = model.predict(X_test, num_iteration=model.best_iteration)
        
        print("\n=== THRESHOLD ANALYSIS ===")
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"Threshold: {threshold:.2f} | Precision: {precision:.3f} | "
                  f"Recall: {recall:.3f} | F1: {f1:.3f}")


class ExoplanetClassifierPipeline:
    """Main pipeline class that orchestrates the entire process."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.results: Optional[Dict[str, Any]] = None
    
    def run_full_pipeline(self, df: pd.DataFrame, target_features: int = None) -> Dict[str, Any]:
        """Execute the complete machine learning pipeline."""
        if target_features is None:
            target_features = Config.TARGET_FEATURES
            
        print("=== EXOPLANET CLASSIFICATION PIPELINE ===")
        print(f"Starting pipeline with dataset shape: {df.shape}")
        
        # Phase 1: Data Preprocessing
        print("\n=== PHASE 1: DATA PREPROCESSING ===")
        X_clean, y_clean = self.preprocessor.clean_data(df)
        
        # Phase 2: Feature Engineering
        print("\n=== PHASE 2: FEATURE ENGINEERING ===")
        X_with_features = self.feature_engineer.create_all_features(X_clean)
        
        # Phase 3: Feature Selection
        print("\n=== PHASE 3: FEATURE SELECTION ===")
        X_selected, selected_features, feature_scores = self.feature_selector.combined_selection(
            X_with_features, y_clean, final_k=target_features
        )
        
        # Phase 4: Final Scaling
        print("\n=== PHASE 4: FINAL SCALING ===")
        X_final, _ = self.preprocessor.scale_features(X_selected)
        
        # Phase 5: Model Training
        print("\n=== PHASE 5: MODEL TRAINING ===")
        # Prepare labels (2 -> 1, 1 -> 0 for binary classification)
        y_binary = y_clean.replace({2: 1, 1: 0})
        
        training_results = self.model_trainer.train_with_cv(X_final, y_binary)
        
        # Compile results
        self.results = {
            'X_final': X_final,
            'y_final': y_binary,
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'model_results': training_results,
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer,
            'feature_selector': self.feature_selector,
            'model_trainer': self.model_trainer,
            'original_shape': df.shape,
            'final_shape': X_final.shape
        }
        
        # Print final summary
        self._print_pipeline_summary()
        
        return self.results
    
    def _print_pipeline_summary(self):
        """Print comprehensive pipeline summary."""
        if self.results is None:
            return
            
        print(f"\n=== PIPELINE SUMMARY ===")
        print(f"Original dataset shape: {self.results['original_shape']}")
        print(f"Final dataset shape: {self.results['final_shape']}")
        print(f"Feature reduction: {self.results['original_shape'][1]-1} → {self.results['final_shape'][1]}")
        print(f"Label distribution:\n{self.results['y_final'].value_counts().sort_index()}")
        
        # Test results
        test_results = self.results['model_results']['test_results']
        print(f"\n=== FINAL TEST RESULTS ===")
        for metric, value in test_results.items():
            if metric not in ['confusion_matrix', 'predictions']:
                print(f"{metric.capitalize()}: {value:.4f}")
        print(f"Confusion Matrix:\n{test_results['confusion_matrix']}")
    
    def save_results(self, filepath: str):
        """Save pipeline results to pickle file."""
        if self.results is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
            print(f"✅ Results saved to {filepath}")
        else:
            print("❌ No results to save. Run pipeline first.")
    
    def load_results(self, filepath: str):
        """Load pipeline results from pickle file."""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        print(f"✅ Results loaded from {filepath}")
    
    def predict_on_new_data(self, new_df: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new dataset."""
        if self.results is None:
            raise ValueError("Pipeline must be trained first or results loaded")
        
        # Process new data through the same pipeline
        print("Processing new data through pipeline...")
        
        # Use the same preprocessing steps
        X_clean, _ = self.preprocessor.clean_data(new_df)
        X_with_features = self.feature_engineer.create_all_features(X_clean)
        
        # Select same features
        X_selected = X_with_features[self.results['selected_features']]
        
        # Scale using fitted scaler
        X_final, _ = self.preprocessor.scale_features(X_selected)
        
        # Make predictions
        model = self.results['model_results']['model']
        y_prob = model.predict(X_final, num_iteration=model.best_iteration)
        y_pred = (y_prob >= threshold).astype(int)
        
        return y_pred, y_prob


def main():
    """Main execution function."""
    try:
        # Initialize pipeline
        pipeline = ExoplanetClassifierPipeline()
        
        # Load data
        train_df, test_df = DataLoader.load_train_test_datasets()
        
        # Run pipeline on training data
        results = pipeline.run_full_pipeline(train_df, target_features=Config.TARGET_FEATURES)
        
        # Save results
        pipeline.save_results("exoplanet_pipeline_results.pkl")
        
        # Optional: Plot feature importance
        try:
            pipeline.feature_selector.plot_feature_importance(top_n=20)
        except:
            print("Feature importance plot failed, continuing...")
        
        # Evaluate on external test data if available
        if test_df is not None:
            print("\n=== EXTERNAL TEST EVALUATION ===")
            
            # Process test data and make predictions
            y_pred_external, y_prob_external = pipeline.predict_on_new_data(test_df)
            
            # If test data has labels, evaluate
            if 'LABEL' in test_df.columns:
                y_test_true = test_df['LABEL'].replace({2: 1, 1: 0})
                
                # Evaluate with different thresholds
                model = results['model_results']['model']
                X_test_processed, _ = pipeline.preprocessor.clean_data(test_df)
                X_test_features = pipeline.feature_engineer.create_all_features(X_test_processed)
                X_test_selected = X_test_features[results['selected_features']]
                X_test_final, _ = pipeline.preprocessor.scale_features(X_test_selected)
                
                pipeline.model_trainer.evaluate_with_thresholds(
                    model, X_test_final, y_test_true
                )
                
                # Final evaluation
                print(f"\n=== EXTERNAL TEST RESULTS ===")
                print(f"Accuracy: {accuracy_score(y_test_true, y_pred_external):.4f}")
                print(f"Precision: {precision_score(y_test_true, y_pred_external, zero_division=0):.4f}")
                print(f"Recall: {recall_score(y_test_true, y_pred_external, zero_division=0):.4f}")
                print(f"F1-Score: {f1_score(y_test_true, y_pred_external, zero_division=0):.4f}")
                print(f"Confusion Matrix:\n{confusion_matrix(y_test_true, y_pred_external)}")
            else:
                print("Test data has no labels. Predictions completed.")
                print(f"Predictions distribution: {Counter(y_pred_external)}")
        
        print("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()