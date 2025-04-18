""" 
Crime Prediction Model Training
This module trains machine learning models to predict crime risk using temporal sliding windows.
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from preprocess import load_datasets, create_crime_prediction_dataset
from datetime import datetime, timedelta
import logging
import warnings

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

def plot_feature_importance(model, X, output_path="feature_importance.png"):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Get feature names
        feature_names = X.columns

        # Top 20 features
        top_indices = indices[:20]
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(top_indices)), importances[top_indices], align='center')
        plt.xticks(range(len(top_indices)), [feature_names[i] for i in top_indices], rotation=90)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Feature importance plot saved as {output_path}")
    else:
        logger.warning("Model doesn't support feature importances")

def plot_precision_recall_curve(y_true, y_pred_proba, output_path="precision_recall.png"):
    """Plot precision-recall curve for binary classification"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:0.3f}')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Precision-recall curve saved as {output_path}")

def train_temporal_model(temporal_data, prediction_window=24, use_time_split=True):
    """
    Train a model using temporal sliding window data
    
    Args:
        temporal_data: DataFrame with temporal sliding windows
        prediction_window: The prediction window (hours) to target
        use_time_split: Whether to use time-based cross-validation
        
    Returns:
        The best trained model
    """
    logger.info(f"Training model for {prediction_window}h prediction window")
    
    # Ensure temporal data is sorted by reference time for proper temporal splits
    temporal_data = temporal_data.sort_values('reference_time')
    
    # Define the target variable for this prediction window
    target_col = f'target_{prediction_window}h'
    
    # Check if we have the target column
    if target_col not in temporal_data.columns:
        logger.error(f"Target column {target_col} not found in temporal data")
        raise ValueError(f"Target column {target_col} not found")
    
    logger.info(f"Target distribution: {temporal_data[target_col].value_counts()}")
    
    # Drop unnecessary columns
    drop_cols = ['geometry', 'reference_time'] + [c for c in temporal_data.columns if c.startswith('target_') and c != target_col]
    X = temporal_data.drop(drop_cols, axis=1, errors='ignore')
    y = temporal_data[target_col]
    
    # Record cell_id for later reference (if available)
    id_cols = [col for col in X.columns if col in ['cell_id', 'h3_index']]
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object' and col not in id_cols]
    numerical_cols = [col for col in X.columns if X[col].dtype != 'object' and col not in id_cols]
    
    logger.info(f"Data shape: {X.shape}, Positive cases: {sum(y)}, Negative cases: {len(y) - sum(y)}")
    logger.info(f"Using {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
            ('pass', 'passthrough', id_cols)  # Pass ids through unchanged
        ],
        remainder='drop'
    )
    
    # Split the data - using time-based split or random split
    if use_time_split:
        # Use a time-based split - 80% for training, 20% for testing
        split_idx = int(len(X) * 0.8)
        X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logger.info(f"Using time-based split: {len(X_train_raw)} train, {len(X_test_raw)} test samples")
        
        # For cross-validation, use TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=5)
    else:
        # Use random split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Using random split: {len(X_train_raw)} train, {len(X_test_raw)} test samples")
        
        # For cross-validation, use 5-fold CV
        cv = 5
    
    # Define models with pipelines
    models = {
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        'gradient_boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        'hist_gradient_boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(random_state=42))
        ])
    }
    
    # Parameter grids for each model
    param_grids = {
        'logistic_regression': {
            'classifier__C': [0.01, 0.1, 1, 10]
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        'gradient_boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        },
        'hist_gradient_boosting': {
            'classifier__max_depth': [None, 5, 10],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_iter': [100, 200]
        }
    }
    
    # Train and evaluate each model
    results = {}
    best_models = {}
    
    with warnings.catch_warnings():
        # Ignore convergence warnings from logistic regression
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for model_name, pipeline in models.items():
            logger.info(f"\nTraining {model_name} for {prediction_window}h window...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train_raw, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test_raw)
            y_prob = best_model.predict_proba(X_test_raw)[:, 1]
            
            # Calculate metrics
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_accuracy': best_model.score(X_test_raw, y_test),
                'test_roc_auc': roc_auc_score(y_test, y_prob),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            logger.info(f"Test accuracy: {results[model_name]['test_accuracy']:.4f}")
            logger.info(f"Test ROC AUC: {results[model_name]['test_roc_auc']:.4f}")
            logger.info(f"Classification Report:\n{results[model_name]['classification_report']}")
            
            # Plot precision-recall curve for this model
            plot_precision_recall_curve(
                y_test, 
                y_prob, 
                output_path=f"{model_name}_{prediction_window}h_pr_curve.png"
            )
            
            # Save the model
            model_filename = f"{model_name}_crime_prediction_{prediction_window}h.pkl"
            joblib.dump(best_model, model_filename)
            logger.info(f"Model saved as {model_filename}")
    
    # Find best overall model
    best_model_name = max(results, key=lambda k: results[k]['test_roc_auc'])
    logger.info(f"\nBest overall model for {prediction_window}h window: {best_model_name}")
    best_model = best_models[best_model_name]
    
    # Get feature importance if applicable
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        # Process X to get feature names after preprocessing
        try:
            # Get column names after preprocessing
            cat_encoder = OneHotEncoder(drop='first', sparse_output=False)
            if categorical_cols:
                cat_transformed = cat_encoder.fit_transform(X_train_raw[categorical_cols])
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
            else:
                cat_feature_names = []
            
            # Combine with numerical column names
            feature_names = numerical_cols + cat_feature_names + id_cols
            
            # Create a DataFrame for feature importance
            importances = best_model.named_steps['classifier'].feature_importances_
            
            # Check if lengths match - they might not due to preprocessing
            if len(importances) == len(feature_names):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Save feature importance
                feature_importance.to_csv(f"feature_importance_{prediction_window}h.csv", index=False)
                logger.info(f"Feature importance saved to feature_importance_{prediction_window}h.csv")
                
                # Plot feature importance
                plot_feature_importance(
                    best_model.named_steps['classifier'], 
                    pd.DataFrame(columns=feature_names), 
                    output_path=f"feature_importance_{prediction_window}h.png"
                )
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {e}")
    
    # Save the best model as the default for this time window
    best_filename = f"best_model_crime_prediction_{prediction_window}h.pkl"
    joblib.dump(best_model, best_filename)
    logger.info(f"Best model saved as {best_filename}")
    
    return best_model, results

def main():
    """Main training function with complete preprocessing pipeline"""
    logger.info("Starting crime prediction model training...")
    
    # Load datasets
    logger.info("Loading datasets...")
    grid_gdf, gdf_crimenes, gdf_police, gdf_barrios = load_datasets()
    
    # Define prediction windows
    prediction_windows = [6, 12, 24, 72]  # 6h, 12h, 24h, 3d
    logger.info(f"Training models for prediction windows: {prediction_windows} hours")
    
    # Create the temporal crime prediction dataset
    logger.info("Creating temporal crime prediction dataset...")
    temporal_data = create_crime_prediction_dataset(
        grid_gdf,
        gdf_crimenes,
        gdf_police,
        gdf_barrios,
        temporal_windows=True,
        prediction_windows=prediction_windows
    )
    
    if temporal_data is None or len(temporal_data) == 0:
        logger.error("No temporal data generated. Check your date ranges and crime data.")
        # Fall back to static dataset approach
        logger.info("Falling back to static dataset approach...")
        static_dataset = create_crime_prediction_dataset(
            grid_gdf,
            gdf_crimenes,
            gdf_police,
            gdf_barrios,
            temporal_windows=False,
            prediction_windows=prediction_windows
        )
        
        return train_static_models(static_dataset, prediction_windows)
    
    # Train models for each prediction window
    best_models = {}
    for window in prediction_windows:
        best_model, results = train_temporal_model(temporal_data, prediction_window=window)
        best_models[window] = best_model
    
    logger.info("Training completed successfully")
    
    return best_models

def train_static_models(crime_prediction_dataset, prediction_windows=None):
    """Train models using a static dataset (no temporal windows)"""
    if prediction_windows is None:
        prediction_windows = [6, 12, 24, 72]  # 6h, 12h, 24h, 3d
        
    logger.info("Training models using static dataset approach...")
    
    best_models = {}
    for window in prediction_windows:
        target_col = f'target_{window}h'
        if target_col not in crime_prediction_dataset.columns:
            logger.warning(f"Target column {target_col} not found in dataset. Skipping {window}h window.")
            continue
            
        logger.info(f"Training model for {window}h window using static approach...")
        
        # Extract features and target
        X = crime_prediction_dataset.drop(['geometry'] + 
                              [col for col in crime_prediction_dataset.columns if col.startswith('target_')], 
                              axis=1, errors='ignore')
        y = crime_prediction_dataset[target_col]
        
        # Standard train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # For static data, just train a gradient boosting classifier
        model = HistGradientBoostingClassifier(
            max_iter=200, 
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"Model for {window}h window - Test accuracy: {model.score(X_test, y_test):.4f}")
        logger.info(f"Model for {window}h window - Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save the model
        model_filename = f"static_model_crime_prediction_{window}h.pkl"
        joblib.dump(model, model_filename)
        logger.info(f"Static model saved as {model_filename}")
        
        best_models[window] = model
    
    return best_models

if __name__ == "__main__":
    logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        best_models = main()
        
        # Print summary of models
        logger.info("\nTraining summary:")
        for window, model in best_models.items():
            logger.info(f"Model for {window}h prediction window: {type(model).__name__}")
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
    
    logger.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
