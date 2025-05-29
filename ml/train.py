""" 
Enhanced Crime Prediction Model Training with 7 Models and Cross-Validation
This module trains 7 specific machine learning models to predict crime risk using 12-hour temporal windows.
Implements proper temporal train/validation/test splits and comprehensive evaluation metrics.
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import json
import logging
import warnings
import argparse
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                           precision_recall_curve, average_precision_score, log_loss,
                           precision_score, recall_score, f1_score, make_scorer)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Try to import LightGBM and XGBoost
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce verbosity
except ImportError:
    OPTUNA_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_preprocessed_dataset(dataset_path):
    """
    Load the preprocessed temporal dataset
    
    Args:
        dataset_path: Path to the pickled dataset file
        
    Returns:
        DataFrame with temporal data
    """
    logger.info(f"Loading preprocessed dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    try:
        temporal_data = pd.read_pickle(dataset_path)
        logger.info(f"Successfully loaded dataset with {len(temporal_data)} samples")
        
        # Check for target columns
        target_cols = [col for col in temporal_data.columns if col.startswith('target_')]
        logger.info(f"Found target columns: {target_cols}")
        
        # Log data types for debugging
        logger.info("Dataset column types:")
        for col in temporal_data.columns:
            dtype = temporal_data[col].dtype
            unique_count = len(temporal_data[col].unique())
            unique_vals = unique_count if unique_count < 10 else "many"
            logger.info(f"  {col}: {dtype} (unique values: {unique_vals})")
        
        return temporal_data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def identify_feature_types(data, target_col):
    """
    Properly identify and categorize feature types to avoid preprocessing errors
    
    Args:
        data: DataFrame with features
        target_col: Target column name
    
    Returns:
        Dictionary with categorized feature lists
    """
    logger.info("Identifying feature types...")
    
    # Columns to always exclude
    always_exclude = ['geometry', 'reference_time', 'ref_month'] + [c for c in data.columns if c.startswith('target_')]
    
    # Identifier columns (H3 IDs, cell IDs, etc.) - these are strings but not categorical features
    id_patterns = ['h3', 'cell_id', 'hex', 'id']
    id_cols = []
    for col in data.columns:
        if any(pattern in col.lower() for pattern in id_patterns):
            id_cols.append(col)
    
    # True categorical columns (non-numeric, not IDs, with reasonable cardinality)
    categorical_cols = []
    for col in data.columns:
        if col not in always_exclude and col not in id_cols:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                unique_count = data[col].nunique()
                # Only treat as categorical if it has reasonable cardinality
                if unique_count < 50:  # Adjust threshold as needed
                    categorical_cols.append(col)
                else:
                    logger.warning(f"Column {col} has {unique_count} unique values - treating as identifier")
                    id_cols.append(col)
    
    # Numerical columns (true numeric types)
    numerical_cols = []
    for col in data.columns:
        if col not in always_exclude and col not in id_cols and col not in categorical_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                numerical_cols.append(col)
            else:
                logger.warning(f"Column {col} has non-numeric type {data[col].dtype} but not categorized - excluding")
                always_exclude.append(col)
    
    feature_types = {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'identifiers': id_cols,
        'exclude': always_exclude
    }
    
    logger.info(f"Feature categorization:")
    logger.info(f"  Numerical features ({len(numerical_cols)}): {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
    logger.info(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"  Identifier columns ({len(id_cols)}): {id_cols}")
    logger.info(f"  Excluded columns ({len(always_exclude)}): {[c for c in always_exclude if not c.startswith('target_')]}")
    
    return feature_types

def create_temporal_splits(temporal_data, train_end_month=9, val_end_month=11):
    """
    Create temporal train/validation/test splits based on months
    
    Args:
        temporal_data: DataFrame with temporal data including 'reference_time'
        train_end_month: Last month for training (1-12), default 9 (September)
        val_end_month: Last month for validation (1-12), default 11 (November)
    
    Returns:
        train_data, val_data, test_data
    """
    logger.info("Creating temporal train/validation/test splits...")
    
    # Ensure reference_time is datetime
    temporal_data['reference_time'] = pd.to_datetime(temporal_data['reference_time'])
    
    # Extract month from reference_time
    temporal_data['ref_month'] = temporal_data['reference_time'].dt.month
    
    # Create splits
    train_data = temporal_data[temporal_data['ref_month'] <= train_end_month].copy()
    val_data = temporal_data[
        (temporal_data['ref_month'] > train_end_month) & 
        (temporal_data['ref_month'] <= val_end_month)
    ].copy()
    test_data = temporal_data[temporal_data['ref_month'] > val_end_month].copy()
    
    logger.info(f"Temporal splits created:")
    logger.info(f"  Training (70%): {len(train_data)} samples (Jan-{train_end_month})")
    logger.info(f"  Validation (15%): {len(val_data)} samples ({train_end_month+1}-{val_end_month})")
    logger.info(f"  Test (15%): {len(test_data)} samples ({val_end_month+1}-Dec)")
    
    # Check class distributions
    for name, data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
        target_cols = [col for col in data.columns if col.startswith('target_')]
        for target_col in target_cols:
            if target_col in data.columns:
                pos_count = data[target_col].sum()
                total = len(data)
                logger.info(f"  {name} - {target_col}: {pos_count}/{total} ({pos_count/total:.2%})")
    
    return train_data, val_data, test_data

def perform_cross_validation(pipeline, X_train, y_train, cv_folds=5, scoring_metrics=None):
    """
    Perform cross-validation on training data
    
    Args:
        pipeline: Scikit-learn pipeline
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
        scoring_metrics: List of scoring metrics
        
    Returns:
        Dictionary with CV scores
    """
    if scoring_metrics is None:
        scoring_metrics = ['roc_auc', 'average_precision', 'neg_log_loss', 'precision']
    
    cv_results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    logger.info(f"Performing {cv_folds}-fold cross-validation...")
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")
            cv_results[metric] = {'mean': 0.0, 'std': 0.0, 'scores': []}
    
    return cv_results

def plot_feature_importance(model, feature_names, output_path="feature_importance.png", top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Top N features
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(len(top_indices)), importances[top_indices], align='center')
        plt.xticks(range(len(top_indices)), [feature_names[i] for i in top_indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved as {output_path}")
        
        # Return top features for analysis
        top_features = [(feature_names[i], importances[i]) for i in top_indices]
        return top_features
    else:
        logger.warning("Model doesn't support feature importances")
        return None

def plot_precision_recall_curve(y_true, y_pred_proba, model_name, output_path="precision_recall.png"):
    """Plot precision-recall curve for binary classification"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.8, where='post', linewidth=2)
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{model_name} - Precision-Recall Curve\nAP = {average_precision:.3f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Precision-recall curve saved as {output_path}")

def evaluate_model_comprehensive(model, X_test, y_test, model_name, output_dir="model_evaluation"):
    """
    Comprehensive model evaluation with all required metrics focusing on probability outputs
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for output files
        output_dir: Directory to save evaluation plots
    
    Returns:
        Dictionary with all evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions and probabilities (risk scores)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Risk scores (probabilities)
    
    # Calculate all required metrics
    metrics = {
        'model_name': model_name,
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'accuracy': (y_pred == y_test).mean(),
        'risk_scores_mean': y_pred_proba.mean(),
        'risk_scores_std': y_pred_proba.std()
    }
    
    # Print comprehensive results
    logger.info(f"\n=== {model_name} Evaluation Results ===")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Risk Scores - Mean: {metrics['risk_scores_mean']:.4f}, Std: {metrics['risk_scores_std']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Classification report
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Plot precision-recall curve
    pr_path = os.path.join(output_dir, f"{model_name}_precision_recall.png")
    plot_precision_recall_curve(y_test, y_pred_proba, model_name, pr_path)
    
    return metrics

def create_model_pipelines(preprocessor):
    """
    Create the 7 required model pipelines: 4 supervised + 3 ensemble
    Optimized for faster training while maintaining quality
    
    Args:
        preprocessor: Sklearn preprocessing pipeline
        
    Returns:
        Dictionary of model pipelines
    """
    logger.info("Creating 7 optimized model pipelines...")
    
    # Base models with optimized parameters for faster training
    base_models = {
        # 4 Supervised Models
        'logistic_regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced',
            solver='liblinear'  # Faster for small datasets
        ),
        'svm': CalibratedClassifierCV(
            estimator=LinearSVC(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                tol=1e-3,
                C=1.0
            ),
            method='sigmoid',
            cv=5
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=1  # Limit parallelization to avoid conflicts
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        )
    }
    
    # Add ensemble models if available
    if LIGHTGBM_AVAILABLE:
        base_models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbosity=-1,
            n_estimators=50,  # Reduced for faster training
            num_leaves=31,
            learning_rate=0.1
        )
    else:
        logger.warning("LightGBM not available - install with: pip install lightgbm")
    
    if XGBOOST_AVAILABLE:
        base_models['xgboost'] = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_estimators=50,  # Reduced for faster training
            max_depth=6,
            learning_rate=0.1
        )
    else:
        logger.warning("XGBoost not available - install with: pip install xgboost")
    
    # Create pipelines
    pipelines = {}
    
    for name, model in base_models.items():
        pipelines[name] = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    # Create Stacking Classifier (Smart Ensemble) - use faster base models
    if len(base_models) >= 3:
        # Use fastest 3 models as base estimators for stacking
        fast_base_estimators = [
            ('logistic', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')),
            ('tree', DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced')),
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance'))
        ]
        
        stacking_classifier = StackingClassifier(
            estimators=fast_base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,
            stack_method='predict_proba',  # Use probabilities for stacking
            n_jobs=1  # Limit parallelization
        )
        
        pipelines['stacking_classifier'] = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', stacking_classifier)
        ])
    
    logger.info(f"Created {len(pipelines)} optimized model pipelines:")
    for name in pipelines.keys():
        logger.info(f"  - {name}")
    
    return pipelines

def train_models_with_cv(temporal_data, prediction_window=12, dataset_path=None):
    """
    Train 7 models with cross-validation using preprocessed data
    
    Args:
        temporal_data: DataFrame with temporal sliding windows (if None, load from dataset_path)
        prediction_window: The prediction window (hours) to target
        dataset_path: Path to preprocessed dataset (if temporal_data is None)
        
    Returns:
        Dictionary with trained models and evaluation results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ENHANCED TRAINING: 7 MODELS WITH CROSS-VALIDATION")
    logger.info(f"Prediction Window: {prediction_window} hours")
    logger.info(f"{'='*80}")
    
    # Load data if not provided
    if temporal_data is None:
        if dataset_path is None:
            dataset_path = f"crime_prediction_temporal_dataset_{prediction_window}h.pkl"
        temporal_data = load_preprocessed_dataset(dataset_path)
    
    # Create temporal splits (70% train, 15% val, 15% test) - DONE ONLY ONCE HERE
    train_data, val_data, test_data = create_temporal_splits(temporal_data)
    
    # Define the target variable for this prediction window
    target_col = f'target_{prediction_window}h'
    
    # Check if we have the target column
    if target_col not in temporal_data.columns:
        logger.error(f"Target column {target_col} not found in temporal data")
        logger.info(f"Available target columns: {[col for col in temporal_data.columns if col.startswith('target_')]}")
        raise ValueError(f"Target column {target_col} not found")
    
    # Identify feature types properly
    feature_types = identify_feature_types(train_data, target_col)
    
    # Prepare features and targets for each split
    X_train_raw = train_data.drop(feature_types['exclude'], axis=1, errors='ignore')
    y_train = train_data[target_col]
    
    X_val_raw = val_data.drop(feature_types['exclude'], axis=1, errors='ignore')
    y_val = val_data[target_col]
    
    X_test_raw = test_data.drop(feature_types['exclude'], axis=1, errors='ignore')
    y_test = test_data[target_col]
    
    # Separate H3 indices from training features
    X_train, h3_train = separate_h3_indices(X_train_raw, feature_types)
    X_val, h3_val = separate_h3_indices(X_val_raw, feature_types)
    X_test, h3_test = separate_h3_indices(X_test_raw, feature_types)
    
    logger.info(f"Data shapes after H3 separation - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Using {len(feature_types['numerical'])} numerical features and {len(feature_types['categorical'])} categorical features")
    
    # Create preprocessing pipeline with proper feature handling
    transformers = []
    
    # Handle numerical features
    if feature_types['numerical']:
        num_features = [col for col in feature_types['numerical'] if col in X_train.columns]
        if num_features:
            transformers.append(('num', StandardScaler(), num_features))
    
    # Handle categorical features
    if feature_types['categorical']:
        cat_features = [col for col in feature_types['categorical'] if col in X_train.columns]
        if cat_features:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features))
    
    # Handle remaining identifier columns (excluding H3 which was separated)
    remaining_ids = [col for col in feature_types['identifiers'] if col in X_train.columns]
    if remaining_ids:
        useful_ids = []
        for id_col in remaining_ids:
            unique_count = X_train[id_col].nunique()
            if unique_count < 100:  # Lower threshold for useful identifiers
                useful_ids.append(id_col)
                logger.info(f"Including identifier {id_col} with {unique_count} unique values")
            else:
                logger.info(f"Excluding identifier {id_col} with {unique_count} unique values (too many)")
        
        if useful_ids:
            transformers.append(('id', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), useful_ids))
    
    # Create the preprocessor
    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any remaining columns
        )
    else:
        logger.error("No valid features found for preprocessing!")
        raise ValueError("No valid features found for preprocessing!")
    
    # Fit preprocessor on training data and transform all splits
    logger.info("Preprocessing features...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE only to training data (70%)
    X_train_final, y_train_final = apply_smote_to_training(X_train_processed, y_train)
    
    # For validation and test, use original processed data
    X_val_final = X_val_processed
    y_val_final = y_val
    X_test_final = X_test_processed
    y_test_final = y_test
    
    # Export final training data for review
    export_final_training_data(
        X_train_final, y_train_final, 
        X_val_final, y_val_final, 
        X_test_final, y_test_final,
        h3_train, h3_val, h3_test,
        prediction_window
    )
    
    # Create model pipelines (without preprocessing since data is already processed)
    logger.info("Creating model pipelines for preprocessed data...")
    
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced',
            solver='liblinear'
        ),
        'svm': CalibratedClassifierCV(
            estimator=LinearSVC(
                random_state=42,
                class_weight='balanced',
                max_iter=1000,
                tol=1e-3,
                C=1.0
            ),
            method='sigmoid',
            cv=5
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=1
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10
        )
    }
    
    # Add ensemble models if available
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbosity=-1,
            n_estimators=50,
            num_leaves=31,
            learning_rate=0.1
        )
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1
        )
    
    # Add stacking classifier
    if len(models) >= 3:
        fast_base_estimators = [
            ('logistic', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')),
            ('tree', DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced')),
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance'))
        ]
        
        models['stacking_classifier'] = StackingClassifier(
            estimators=fast_base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,
            stack_method='predict_proba',
            n_jobs=1
        )
    
    # Train and evaluate each model
    results = {}
    trained_models = {}
    all_metrics = []
    cv_results_all = {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name.upper()}...")
            logger.info(f"{'='*60}")
            
            try:
                # Perform cross-validation on processed training data
                cv_results = perform_cross_validation(model, X_train_final, y_train_final)
                cv_results_all[model_name] = cv_results
                
                # Train the model on processed training set
                model.fit(X_train_final, y_train_final)
                trained_models[model_name] = model
                
                # Evaluate on validation set for model selection
                val_metrics = evaluate_model_comprehensive(
                    model, X_val_final, y_val_final, f"{model_name}_validation"
                )
                val_metrics['cv_roc_auc_mean'] = cv_results.get('roc_auc', {}).get('mean', 0.0)
                val_metrics['cv_roc_auc_std'] = cv_results.get('roc_auc', {}).get('std', 0.0)
                
                # Evaluate on test set for final assessment
                test_metrics = evaluate_model_comprehensive(
                    model, X_test_final, y_test_final, f"{model_name}_test"
                )
                test_metrics['cv_roc_auc_mean'] = cv_results.get('roc_auc', {}).get('mean', 0.0)
                test_metrics['cv_roc_auc_std'] = cv_results.get('roc_auc', {}).get('std', 0.0)
                
                # Store results
                results[model_name] = {
                    'validation_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'cv_results': cv_results,
                    'model': model,
                    'preprocessor': preprocessor,
                    'h3_mapping': {
                        'train': h3_train,
                        'val': h3_val,
                        'test': h3_test
                    }
                }
                
                # Add to metrics list for comparison
                test_metrics['split'] = 'test'
                test_metrics['model'] = model_name
                all_metrics.append(test_metrics)
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    try:
                        # Get feature names after preprocessing
                        feature_names_out = preprocessor.get_feature_names_out()
                        
                        # Plot feature importance
                        importance_path = f"feature_importance_{model_name}_{prediction_window}h.png"
                        top_features = plot_feature_importance(
                            model,
                            feature_names_out,
                            importance_path
                        )
                        
                        if top_features:
                            logger.info(f"Top 5 features for {model_name}:")
                            for i, (feature, importance) in enumerate(top_features[:5]):
                                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
                                
                    except Exception as e:
                        logger.warning(f"Could not generate feature importance for {model_name}: {e}")
                
                # Save the model with preprocessing info
                model_data = {
                    'model': model,
                    'preprocessor': preprocessor,
                    'feature_types': feature_types,
                    'prediction_window': prediction_window,
                    'training_date': datetime.now().isoformat(),
                    'use_smote': True,
                    'metrics': {
                        'validation': val_metrics,
                        'test': test_metrics,
                        'cv': cv_results
                    }
                }
                
                model_filename = f"{model_name}_crime_prediction_{prediction_window}h.pkl"
                joblib.dump(model_data, model_filename)
                logger.info(f"Model saved as {model_filename}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                logger.exception("Full traceback:")
                continue
    
    # Perform statistical testing on top 3 models
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL TESTING OF TOP 3 MODELS")
    logger.info("="*80)
    
    # Test both ROC AUC and PR AUC
    for metric in ['roc_auc', 'average_precision']:
        statistical_results = perform_statistical_testing(cv_results_all, metric)
        if statistical_results:
            logger.info(f"Statistical testing completed for {metric}")
    
    # Create comprehensive metrics comparison DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"enhanced_model_comparison_{prediction_window}h.csv", index=False)
    logger.info(f"Enhanced model comparison saved to enhanced_model_comparison_{prediction_window}h.csv")
    
    # Create cross-validation results summary
    cv_summary = []
    for model_name, cv_res in cv_results_all.items():
        cv_row = {'model': model_name}
        for metric, values in cv_res.items():
            cv_row[f'cv_{metric}_mean'] = values['mean']
            cv_row[f'cv_{metric}_std'] = values['std']
        cv_summary.append(cv_row)
    
    cv_df = pd.DataFrame(cv_summary)
    cv_df.to_csv(f"cross_validation_results_{prediction_window}h.csv", index=False)
    logger.info(f"Cross-validation results saved to cross_validation_results_{prediction_window}h.csv")
    
    # Find best model based on validation PR AUC
    best_model_name = max(results.keys(), 
                         key=lambda k: results[k]['validation_metrics']['pr_auc'])
    
    logger.info(f"\n{'='*80}")
    logger.info(f"BEST MODEL for {prediction_window}h: {best_model_name.upper()}")
    logger.info(f"Validation PR AUC: {results[best_model_name]['validation_metrics']['pr_auc']:.4f}")
    logger.info(f"Test PR AUC: {results[best_model_name]['test_metrics']['pr_auc']:.4f}")
    logger.info(f"CV ROC AUC: {results[best_model_name]['cv_results'].get('roc_auc', {}).get('mean', 0.0):.4f} "+
                f"(+/- {results[best_model_name]['cv_results'].get('roc_auc', {}).get('std', 0.0):.4f})")
    logger.info(f"{'='*80}")
    
    # Print summary table
    logger.info(f"\n=== MODEL PERFORMANCE SUMMARY ({prediction_window}h) ===")
    logger.info(f"{'Model':<20} {'Val PR AUC':<12} {'Test PR AUC':<12} {'CV ROC AUC':<12} {'Log Loss':<10}")
    logger.info("-" * 70)
    
    for model_name in results.keys():
        val_pr = results[model_name]['validation_metrics']['pr_auc']
        test_pr = results[model_name]['test_metrics']['pr_auc']
        cv_roc = results[model_name]['cv_results'].get('roc_auc', {}).get('mean', 0.0)
        test_ll = results[model_name]['test_metrics']['log_loss']
        
        logger.info(f"{model_name:<20} {val_pr:<12.4f} {test_pr:<12.4f} {cv_roc:<12.4f} {test_ll:<10.4f}")
    
    best_model = results[best_model_name]['model']
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'all_results': results,
        'metrics_comparison': metrics_df,
        'cv_results': cv_results_all,
        'preprocessor': preprocessor,
        'statistical_tests': cv_results_all  # Include for statistical testing
    }

def separate_h3_indices(X_data, feature_types):
    """
    Separate H3 indices from training features while keeping them for mapping
    
    Args:
        X_data: DataFrame with features
        feature_types: Dictionary with feature categorization
    
    Returns:
        X_features: DataFrame with only training features
        h3_indices: Series with H3 indices for mapping
    """
    logger.info("Separating H3 indices from training features...")
    
    # Find H3 index columns
    h3_cols = [col for col in feature_types['identifiers'] if 'h3' in col.lower()]
    
    if h3_cols:
        # Keep H3 indices separately
        h3_indices = X_data[h3_cols[0]] if len(h3_cols) > 0 else None
        logger.info(f"Separated H3 index column: {h3_cols[0]} ({len(h3_indices.unique())} unique values)")
        
        # Remove H3 indices from features
        X_features = X_data.drop(h3_cols, axis=1, errors='ignore')
        
        # Update feature types to exclude H3 indices from identifiers
        feature_types['identifiers'] = [col for col in feature_types['identifiers'] if col not in h3_cols]
        feature_types['exclude'].extend(h3_cols)
        
        logger.info(f"Features shape after H3 removal: {X_features.shape}")
    else:
        logger.warning("No H3 index columns found")
        h3_indices = None
        X_features = X_data.copy()
    
    return X_features, h3_indices

def apply_smote_to_training(X_train, y_train):
    """
    Apply SMOTE only to the 70% training data
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        X_train_resampled: Resampled training features
        y_train_resampled: Resampled training labels
    """
    logger.info("Applying SMOTE to training data only...")
    
    # Check class distribution before SMOTE
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    logger.info(f"Before SMOTE - Positive: {pos_count}, Negative: {neg_count} (ratio: {pos_count/neg_count:.3f})")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(5, pos_count-1) if pos_count > 1 else 1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    pos_count_after = y_train_resampled.sum()
    neg_count_after = len(y_train_resampled) - pos_count_after
    logger.info(f"After SMOTE - Positive: {pos_count_after}, Negative: {neg_count_after} (ratio: {pos_count_after/neg_count_after:.3f})")
    logger.info(f"Training data increased from {len(y_train)} to {len(y_train_resampled)} samples")
    
    return X_train_resampled, y_train_resampled

def export_final_training_data(X_train_final, y_train_final, X_val_final, y_val_final, X_test_final, y_test_final, 
                               h3_train, h3_val, h3_test, prediction_window):
    """
    Export the final preprocessed data that goes into training
    
    Args:
        X_train_final: Final training features after preprocessing (may be SMOTE-augmented)
        y_train_final: Final training labels (may be SMOTE-augmented)
        X_val_final: Final validation features after preprocessing  
        y_val_final: Final validation labels
        X_test_final: Final test features after preprocessing
        y_test_final: Final test labels
        h3_train, h3_val, h3_test: H3 indices for each split (original size, before SMOTE)
        prediction_window: Prediction window in hours
    """
    logger.info("Exporting final training data for review...")
    
    # Create export directory
    export_dir = f"final_training_data_{prediction_window}h"
    os.makedirs(export_dir, exist_ok=True)
    
    # Export training data (note: after SMOTE, we can't match H3 indices to synthetic samples)
    train_df = pd.DataFrame(X_train_final)
    train_df[f'target_{prediction_window}h'] = y_train_final
    
    # For training data, note that H3 indices don't apply to synthetic SMOTE samples
    if h3_train is not None and len(h3_train) != len(train_df):
        logger.info(f"Training data was augmented with SMOTE ({len(h3_train)} -> {len(train_df)} samples)")
        logger.info("H3 indices not included in training export due to SMOTE augmentation")
        train_df['note'] = 'Original samples: 0-' + str(len(h3_train)-1) + ', Synthetic samples: ' + str(len(h3_train)) + '-' + str(len(train_df)-1)
    elif h3_train is not None:
        train_df['h3_index'] = h3_train.values if hasattr(h3_train, 'values') else h3_train
    
    train_df.to_csv(os.path.join(export_dir, 'train_final.csv'), index=False)
    
    # Export validation data
    val_df = pd.DataFrame(X_val_final)
    val_df[f'target_{prediction_window}h'] = y_val_final
    if h3_val is not None:
        val_df['h3_index'] = h3_val.values if hasattr(h3_val, 'values') else h3_val
    val_df.to_csv(os.path.join(export_dir, 'val_final.csv'), index=False)
    
    # Export test data
    test_df = pd.DataFrame(X_test_final)
    test_df[f'target_{prediction_window}h'] = y_test_final
    if h3_test is not None:
        test_df['h3_index'] = h3_test.values if hasattr(h3_test, 'values') else h3_test
    test_df.to_csv(os.path.join(export_dir, 'test_final.csv'), index=False)
    
    # Export feature names and information
    feature_info = {
        'feature_names': [col for col in train_df.columns if col not in [f'target_{prediction_window}h', 'h3_index', 'note']],
        'n_features': X_train_final.shape[1],
        'train_shape': X_train_final.shape,
        'val_shape': X_val_final.shape,
        'test_shape': X_test_final.shape,
        'smote_applied': len(h3_train) != len(train_df) if h3_train is not None else False,
        'original_train_size': len(h3_train) if h3_train is not None else None,
        'augmented_train_size': len(train_df),
        'class_distribution_train': {
            'positive': int(y_train_final.sum()),
            'negative': int(len(y_train_final) - y_train_final.sum()),
            'ratio': float(y_train_final.mean())
        }
    }
    
    with open(os.path.join(export_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info(f"Final training data exported to {export_dir}/")
    logger.info(f"  - train_final.csv: {train_df.shape}")
    logger.info(f"  - val_final.csv: {val_df.shape}")
    logger.info(f"  - test_final.csv: {test_df.shape}")
    logger.info(f"  - feature_info.json: {len(feature_info['feature_names'])} features")
    
    if feature_info['smote_applied']:
        logger.info(f"  - Note: Training data includes {feature_info['augmented_train_size'] - feature_info['original_train_size']} synthetic SMOTE samples")

def perform_statistical_testing(cv_results_all, metric='roc_auc'):
    """
    Perform ANOVA + Tukey HSD test for top 3 models to determine statistical significance
    
    Args:
        cv_results_all: Dictionary with cross-validation results for all models
        metric: Metric to use for comparison (default: 'roc_auc')
    
    Returns:
        Dictionary with statistical test results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STATISTICAL TESTING: ANOVA + TUKEY HSD")
    logger.info(f"Metric: {metric}")
    logger.info(f"{'='*60}")
    
    # Extract CV scores for the specified metric
    model_scores = {}
    for model_name, cv_results in cv_results_all.items():
        if metric in cv_results and len(cv_results[metric]['scores']) > 0:
            model_scores[model_name] = cv_results[metric]['scores']
    
    if len(model_scores) < 3:
        logger.warning(f"Not enough models with {metric} scores for statistical testing")
        return None
    
    # Find top 3 models based on mean CV score
    mean_scores = {name: np.mean(scores) for name, scores in model_scores.items()}
    top_3_models = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    logger.info(f"Top 3 models by mean {metric}:")
    for i, (model, score) in enumerate(top_3_models):
        logger.info(f"  {i+1}. {model}: {score:.4f}")
    
    # Prepare data for ANOVA
    top_3_names = [model for model, _ in top_3_models]
    top_3_scores = [model_scores[model] for model in top_3_names]
    
    # Perform ANOVA
    f_stat, p_value_anova = f_oneway(*top_3_scores)
    
    logger.info(f"\nANOVA Results:")
    logger.info(f"  F-statistic: {f_stat:.4f}")
    logger.info(f"  p-value: {p_value_anova:.6f}")
    logger.info(f"  Significant difference: {'Yes' if p_value_anova < 0.05 else 'No'}")
    
    # If ANOVA is significant, perform Tukey HSD
    tukey_results = None
    if p_value_anova < 0.05:
        logger.info(f"\nPerforming Tukey HSD post-hoc test...")
        
        # Prepare data for Tukey HSD
        all_scores = []
        all_groups = []
        for model_name in top_3_names:
            scores = model_scores[model_name]
            all_scores.extend(scores)
            all_groups.extend([model_name] * len(scores))
        
        # Perform Tukey HSD
        tukey = pairwise_tukeyhsd(endog=all_scores, groups=all_groups, alpha=0.05)
        
        logger.info(f"Tukey HSD Results:")
        logger.info(str(tukey))
        
        # Extract pairwise comparison results
        tukey_results = {
            'summary': str(tukey),
            'pairwise_comparisons': []
        }
        
        for i in range(len(tukey.groupsunique)):
            for j in range(i+1, len(tukey.groupsunique)):
                group1 = tukey.groupsunique[i]
                group2 = tukey.groupsunique[j]
                # Find the corresponding p-value in the results
                tukey_results['pairwise_comparisons'].append({
                    'group1': group1,
                    'group2': group2,
                    'significant': 'TBD'  # Will be filled from tukey object
                })
    else:
        logger.info(f"\nNo significant differences found - skipping post-hoc test")
    
    # Create statistical summary
    statistical_results = {
        'metric': metric,
        'top_3_models': top_3_names,
        'mean_scores': {name: mean_scores[name] for name in top_3_names},
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value_anova,
            'significant': p_value_anova < 0.05
        },
        'tukey_hsd': tukey_results
    }
    
    # Save statistical results
    with open(f'statistical_testing_results_{metric}.json', 'w') as f:
        json.dump(statistical_results, f, indent=2, default=str)
    
    logger.info(f"\nStatistical testing results saved to statistical_testing_results_{metric}.json")
    
    return statistical_results

def optimize_hyperparameters_optuna(model_name, X_train, y_train, X_val, y_val, n_trials=20):
    """
    Optimize hyperparameters using Optuna for the top 3 models
    
    Args:
        model_name: Name of the model to optimize
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of Optuna trials (default: 20 for faster execution)
    
    Returns:
        Dictionary with best parameters and optimized model
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available - install with: pip install optuna")
        return None
    
    logger.info(f"Optimizing hyperparameters for {model_name} with {n_trials} trials...")
    
    def objective(trial):
        """Objective function for Optuna optimization"""
        
        if model_name == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                'max_iter': trial.suggest_int('max_iter', 500, 2000),
                'class_weight': 'balanced',
                'random_state': 42
            }
            model = LogisticRegression(**params)
            
        elif model_name == 'decision_tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': 42
            }
            model = DecisionTreeClassifier(**params)
            
        elif model_name == 'knn':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
            }
            model = KNeighborsClassifier(**params)
            
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'class_weight': 'balanced',
                'verbosity': -1,
                'random_state': 42
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'eval_metric': 'logloss',
                'verbosity': 0,
                'random_state': 42
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_name == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 500, 2000),
                'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True),
                'class_weight': 'balanced',
                'random_state': 42
            }
            svm_model = LinearSVC(**params)
            model = CalibratedClassifierCV(estimator=svm_model, method='sigmoid', cv=3)
            
        else:
            raise ValueError(f"Model {model_name} not supported for hyperparameter optimization")
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Use PR AUC as optimization metric (better for imbalanced data)
        score = average_precision_score(y_val, y_pred_proba)
        return score
    
    # Create study
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler(seed=42))
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best hyperparameters for {model_name}:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Best validation PR AUC: {best_score:.4f}")
    
    # Create optimized model
    if model_name == 'logistic_regression':
        optimized_model = LogisticRegression(**best_params)
    elif model_name == 'decision_tree':
        optimized_model = DecisionTreeClassifier(**best_params)
    elif model_name == 'knn':
        optimized_model = KNeighborsClassifier(**best_params)
    elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
        optimized_model = lgb.LGBMClassifier(**best_params)
    elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
        optimized_model = xgb.XGBClassifier(**best_params)
    elif model_name == 'svm':
        svm_params = {k: v for k, v in best_params.items() if k != 'method'}
        svm_model = LinearSVC(**svm_params)
        optimized_model = CalibratedClassifierCV(estimator=svm_model, method='sigmoid', cv=3)
    
    # Train optimized model
    optimized_model.fit(X_train, y_train)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'optimized_model': optimized_model,
        'study': study
    }

def create_final_pipeline(best_model, preprocessor, model_name, feature_types):
    """
    Create a single scikit-learn Pipeline that includes all preprocessing steps and the best model
    
    Args:
        best_model: The best trained model
        preprocessor: The fitted preprocessor
        model_name: Name of the best model
        feature_types: Dictionary with feature type information
    
    Returns:
        Complete pipeline ready for deployment
    """
    logger.info(f"Creating final deployment pipeline with {model_name}...")
    
    # Create a new preprocessor (unfitted) with the same configuration
    transformers = []
    
    # Handle numerical features
    if feature_types['numerical']:
        transformers.append(('num', StandardScaler(), feature_types['numerical']))
    
    # Handle categorical features
    if feature_types['categorical']:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), feature_types['categorical']))
    
    # Handle useful identifier columns (excluding H3)
    useful_ids = [col for col in feature_types['identifiers'] if 'h3' not in col.lower()]
    if useful_ids:
        transformers.append(('id', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), useful_ids))
    
    # Create new preprocessor
    new_preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Create the complete pipeline
    final_pipeline = Pipeline([
        ('preprocessor', new_preprocessor),
        ('classifier', best_model)
    ])
    
    logger.info("Final pipeline structure:")
    logger.info(f"  1. Preprocessor with {len(transformers)} transformer groups")
    logger.info(f"  2. Classifier: {model_name}")
    
    return final_pipeline

def hyperparameter_optimization_phase(results, X_train_final, y_train_final, X_val_final, y_val_final, 
                                     preprocessor, feature_types, prediction_window):
    """
    Perform hyperparameter optimization on the top 3 models
    
    Args:
        results: Training results from initial phase
        X_train_final: Final training features
        y_train_final: Final training labels  
        X_val_final: Final validation features
        y_val_final: Final validation labels
        preprocessor: Fitted preprocessor
        feature_types: Feature type information
        prediction_window: Prediction window in hours
    
    Returns:
        Dictionary with optimized models and final pipeline
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"HYPERPARAMETER OPTIMIZATION PHASE")
    logger.info(f"{'='*80}")
    
    # Find top 3 models based on validation PR AUC
    model_rankings = [(name, results[name]['validation_metrics']['pr_auc']) 
                     for name in results.keys()]
    model_rankings.sort(key=lambda x: x[1], reverse=True)
    top_3_models = model_rankings[:3]
    
    logger.info(f"Top 3 models selected for hyperparameter optimization:")
    for i, (model_name, score) in enumerate(top_3_models):
        logger.info(f"  {i+1}. {model_name}: {score:.4f}")
    
    # Optimize each of the top 3 models
    optimized_results = {}
    
    for model_name, _ in top_3_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing {model_name.upper()}...")
        logger.info(f"{'='*60}")
        
        try:
            optimization_result = optimize_hyperparameters_optuna(
                model_name, X_train_final, y_train_final, X_val_final, y_val_final,
                n_trials=20  # Reasonable number for university project
            )
            
            if optimization_result:
                # Evaluate optimized model on validation set
                optimized_model = optimization_result['optimized_model']
                val_metrics = evaluate_model_comprehensive(
                    optimized_model, X_val_final, y_val_final, 
                    f"{model_name}_optimized_validation"
                )
                
                optimized_results[model_name] = {
                    'optimization_result': optimization_result,
                    'validation_metrics': val_metrics,
                    'original_score': results[model_name]['validation_metrics']['pr_auc'],
                    'optimized_score': val_metrics['pr_auc'],
                    'improvement': val_metrics['pr_auc'] - results[model_name]['validation_metrics']['pr_auc']
                }
                
                logger.info(f"{model_name} optimization completed:")
                logger.info(f"  Original PR AUC: {optimized_results[model_name]['original_score']:.4f}")
                logger.info(f"  Optimized PR AUC: {optimized_results[model_name]['optimized_score']:.4f}")
                logger.info(f"  Improvement: {optimized_results[model_name]['improvement']:.4f}")
                
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {e}")
            continue
    
    # Find the best optimized model
    if optimized_results:
        best_optimized_model_name = max(optimized_results.keys(), 
                                       key=lambda k: optimized_results[k]['optimized_score'])
        
        best_optimized_model = optimized_results[best_optimized_model_name]['optimization_result']['optimized_model']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST OPTIMIZED MODEL: {best_optimized_model_name.upper()}")
        logger.info(f"Optimized Validation PR AUC: {optimized_results[best_optimized_model_name]['optimized_score']:.4f}")
        logger.info(f"Improvement over original: {optimized_results[best_optimized_model_name]['improvement']:.4f}")
        logger.info(f"{'='*80}")
        
        # Create final pipeline with best optimized model
        final_pipeline = create_final_pipeline(
            best_optimized_model, preprocessor, best_optimized_model_name, feature_types
        )
        
        # Save optimized model and pipeline
        optimized_model_data = {
            'model': best_optimized_model,
            'pipeline': final_pipeline,
            'preprocessor': preprocessor,
            'feature_types': feature_types,
            'model_name': best_optimized_model_name,
            'prediction_window': prediction_window,
            'training_date': datetime.now().isoformat(),
            'hyperparameters': optimized_results[best_optimized_model_name]['optimization_result']['best_params'],
            'optimization_score': optimized_results[best_optimized_model_name]['optimized_score'],
            'metrics': optimized_results[best_optimized_model_name]['validation_metrics']
        }
        
        # Save the best optimized model
        best_model_filename = f"best_crime_model_{prediction_window}h.pkl"
        joblib.dump(optimized_model_data, best_model_filename)
        logger.info(f"Best optimized model saved as {best_model_filename}")
        
        # Save optimization summary
        optimization_summary = {
            'top_3_models': [name for name, _ in top_3_models],
            'optimization_results': {
                name: {
                    'original_score': res['original_score'],
                    'optimized_score': res['optimized_score'],
                    'improvement': res['improvement'],
                    'best_params': res['optimization_result']['best_params']
                } for name, res in optimized_results.items()
            },
            'best_model': best_optimized_model_name,
            'best_score': optimized_results[best_optimized_model_name]['optimized_score']
        }
        
        with open(f'hyperparameter_optimization_summary_{prediction_window}h.json', 'w') as f:
            json.dump(optimization_summary, f, indent=2)
        
        logger.info(f"Optimization summary saved to hyperparameter_optimization_summary_{prediction_window}h.json")
        
        return {
            'best_model': best_optimized_model,
            'best_model_name': best_optimized_model_name,
            'final_pipeline': final_pipeline,
            'optimization_results': optimized_results,
            'optimization_summary': optimization_summary
        }
    
    else:
        logger.warning("No models were successfully optimized")
        return None

def main():
    """
    Main training function with command line argument support
    """
    parser = argparse.ArgumentParser(description='Train enhanced crime prediction models with 7 algorithms')
    parser.add_argument('--prediction_window', type=int, default=12, 
                       help='Prediction window in hours (default: 12)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to preprocessed dataset (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Start training
    start_time = datetime.now()
    logger.info(f"Enhanced training started at {start_time}")
    logger.info(f"Configuration: {args.prediction_window}h prediction window")
    
    try:
        # Phase 1: Train all 7 models with cross-validation
        training_results = train_models_with_cv(
            temporal_data=None,
            prediction_window=args.prediction_window,
            dataset_path=args.dataset_path
        )
        
        # Phase 2: Hyperparameter optimization on top 3 models
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 2: HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*80}")
        
        # Load the same data for hyperparameter optimization
        if args.dataset_path is None:
            dataset_path = f"crime_prediction_temporal_dataset_{args.prediction_window}h.pkl"
        else:
            dataset_path = args.dataset_path
            
        temporal_data = load_preprocessed_dataset(dataset_path)
        train_data, val_data, test_data = create_temporal_splits(temporal_data)
        
        target_col = f'target_{args.prediction_window}h'
        feature_types = identify_feature_types(train_data, target_col)
        
        # Prepare data for hyperparameter optimization (same preprocessing)
        X_train_raw = train_data.drop(feature_types['exclude'], axis=1, errors='ignore')
        y_train = train_data[target_col]
        X_val_raw = val_data.drop(feature_types['exclude'], axis=1, errors='ignore')
        y_val = val_data[target_col]
        
        X_train, h3_train = separate_h3_indices(X_train_raw, feature_types)
        X_val, h3_val = separate_h3_indices(X_val_raw, feature_types)
        
        # Use the same preprocessor from training
        preprocessor = training_results['preprocessor']
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Apply SMOTE to training data
        X_train_final, y_train_final = apply_smote_to_training(X_train_processed, y_train)
        X_val_final = X_val_processed
        y_val_final = y_val
        
        # Perform hyperparameter optimization
        optimization_results = hyperparameter_optimization_phase(
            training_results['all_results'], 
            X_train_final, y_train_final, 
            X_val_final, y_val_final,
            preprocessor, feature_types, args.prediction_window
        )
        
        # Training completed
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Duration: {duration}")
        logger.info(f"Best Model: {training_results['best_model_name']}")
        if optimization_results:
            logger.info(f"Best Optimized Model: {optimization_results['best_model_name']}")
        logger.info(f"Total Models Trained: {len(training_results['all_results'])}")
        logger.info(f"{'='*80}")
        
        # Save training summary
        summary = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'prediction_window': args.prediction_window,
            'initial_best_model': training_results['best_model_name'],
            'optimized_best_model': optimization_results['best_model_name'] if optimization_results else None,
            'total_models': len(training_results['all_results']),
            'hyperparameter_optimization': optimization_results is not None
        }
        
        with open(f'training_summary_{args.prediction_window}h.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to training_summary_{args.prediction_window}h.json")
        
        # Return combined results
        final_results = training_results.copy()
        if optimization_results:
            final_results['optimization_results'] = optimization_results
            final_results['final_pipeline'] = optimization_results['final_pipeline']
        
        return final_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        raise
    
    return training_results

if __name__ == "__main__":
    logger.info("Starting enhanced crime prediction model training with 7 models and cross-validation...")
    results = main()
