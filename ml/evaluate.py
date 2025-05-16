'''
Evaluación de modelos para predicción de crimen
'''
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import argparse
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load a trained model from file"""
    logger.info(f"Loading model from {model_path}")
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_test_data(data_path):
    """Load test data from file"""
    logger.info(f"Loading test data from {data_path}")
    try:
        if data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.gpkg'):
            import geopandas as gpd
            df = gpd.read_file(data_path)
        else:
            logger.error(f"Unsupported file format: {data_path}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def evaluate_model(model_data, test_data, target_col='target_24h', output_dir='.'):
    """Evaluate model performance on test data"""
    logger.info("Evaluating model performance...")
    
    if 'model' not in model_data:
        logger.error("Invalid model data - missing 'model' key")
        return
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    numerical_cols = model_data.get('numerical_columns')
    
    # Prepare test data
    logger.info(f"Test data shape: {test_data.shape}")
    
    # Check if target column exists
    if target_col not in test_data.columns:
        logger.error(f"Target column '{target_col}' not found in test data")
        # Try alternatives
        alternatives = [col for col in test_data.columns if col.startswith('target_')]
        if not alternatives:
            alternatives = ['target']
        
        if alternatives and alternatives[0] in test_data.columns:
            target_col = alternatives[0]
            logger.warning(f"Using alternative target column: {target_col}")
        else:
            logger.error("No valid target column found")
            return
    
    # Drop unnecessary columns
    drop_cols = ['geometry'] if 'geometry' in test_data.columns else []
    target_cols = [col for col in test_data.columns if col.startswith('target_') and col != target_col]
    drop_cols.extend(target_cols)
    
    # Extract features and target
    X = test_data.drop(drop_cols + [target_col], axis=1)
    y = test_data[target_col]
    
    # Scale features if needed
    if scaler is not None and numerical_cols is not None:
        scale_cols = [col for col in numerical_cols if col in X.columns]
        if scale_cols:
            logger.info(f"Scaling {len(scale_cols)} numerical columns")
            X[scale_cols] = scaler.transform(X[scale_cols])
    
    # Check for missing/extra columns
    missing_cols = set(model.feature_names_in_) - set(X.columns)
    extra_cols = set(X.columns) - set(model.feature_names_in_)
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        # Add missing columns with zeros
        for col in missing_cols:
            X[col] = 0
    
    if extra_cols:
        logger.warning(f"Extra columns will be ignored: {extra_cols}")
    
    # Ensure columns are in the right order
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    logger.info("\nClassification report:")
    report = classification_report(y, y_pred)
    logger.info("\n" + report)
    
    # Save classification report to file
    with open(os.path.join(output_dir, f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"), 'w') as f:
        f.write(report)
    
    # Plot and save confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_file = os.path.join(output_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(cm_file)
    logger.info(f"Confusion matrix saved to {cm_file}")
    
    # Calculate AUC and plot ROC curve (for binary classification)
    try:
        auc = roc_auc_score(y, y_prob)
        logger.info(f"ROC AUC Score: {auc:.4f}")
        
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        roc_file = os.path.join(output_dir, f"roc_curve_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(roc_file)
        logger.info(f"ROC curve saved to {roc_file}")
        
        # Precision-Recall curve
        avg_precision = average_precision_score(y, y_prob)
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y, y_prob)
        plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        pr_file = os.path.join(output_dir, f"pr_curve_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(pr_file)
        logger.info(f"Precision-Recall curve saved to {pr_file}")
        
    except Exception as e:
        logger.warning(f"Could not calculate AUC or plot curves: {e}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        top_features = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Feature Importance')
        plt.tight_layout()
        fi_file = os.path.join(output_dir, f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(fi_file)
        logger.info(f"Feature importance plot saved to {fi_file}")
    
    logger.info("Evaluation completed")
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'auc': auc if 'auc' in locals() else None
    }

def create_evaluation_report(evaluation_results, model_path, test_data_path, output_path=None):
    """Create a detailed evaluation report"""
    if output_path is None:
        output_path = f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    
    logger.info(f"Creating evaluation report: {output_path}")
    
    html = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metric {{ margin-bottom: 15px; }}
            pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            .container {{ max-width: 1000px; margin: auto; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Crime Prediction Model Evaluation Report</h1>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Model:</strong> {model_path}</p>
            <p><strong>Test Data:</strong> {test_data_path}</p>
            
            <h2>Classification Report</h2>
            <pre>{evaluation_results['classification_report']}</pre>
            
            <h2>Performance Metrics</h2>
            <div class="metric">
                <p><strong>ROC AUC Score:</strong> {evaluation_results.get('auc', 'Not available')}</p>
            </div>
            
            <h2>Confusion Matrix</h2>
            <img src="confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.png" alt="Confusion Matrix">
            
            <h2>ROC Curve</h2>
            <img src="roc_curve_{datetime.now().strftime('%Y%m%d_%H%M')}.png" alt="ROC Curve">
            
            <h2>Precision-Recall Curve</h2>
            <img src="pr_curve_{datetime.now().strftime('%Y%m%d_%H%M')}.png" alt="Precision-Recall Curve">
            
            <h2>Feature Importance</h2>
            <img src="feature_importance_{datetime.now().strftime('%Y%m%d_%H%M')}.png" alt="Feature Importance">
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Evaluation report saved to {output_path}")
    return output_path

def evaluate_quick_model(model_path=None, test_data_path=None, target_col='target_24h', output_dir='.'):
    """Convenience function to evaluate the quick model"""
    # Try to find the model if not provided
    if model_path is None:
        if os.path.exists('crime_model_quick.pkl'):
            model_path = 'crime_model_quick.pkl'
        elif os.path.exists(f'crime_model_quick_{target_col}.pkl'):
            model_path = f'crime_model_quick_{target_col}.pkl'
        else:
            logger.error("No quick model found. Run quick_train.py first.")
            return None
    
    # Load the model
    model_data = load_model(model_path)
    if model_data is None:
        return None
    
    # Try to find test data if not provided
    if test_data_path is None:
        if os.path.exists('crime_prediction_temporal_dataset.pkl'):
            test_data_path = 'crime_prediction_temporal_dataset.pkl'
        elif os.path.exists('crime_prediction_grid.gpkg'):
            test_data_path = 'crime_prediction_grid.gpkg'
        else:
            logger.error("No dataset found. Run preprocess.py first.")
            return None
    
    # Load test data
    test_data = load_test_data(test_data_path)
    if test_data is None:
        return None
    
    # If the dataset is large, take a subset to speed up evaluation
    if len(test_data) > 10000:
        logger.info(f"Using a random subset of 10,000 samples for evaluation (from {len(test_data)} total)")
        test_data = test_data.sample(10000, random_state=42)
    
    # Evaluate model
    evaluation_results = evaluate_model(model_data, test_data, target_col, output_dir)
    
    # Create a report
    if evaluation_results:
        report_path = create_evaluation_report(evaluation_results, model_path, test_data_path)
        logger.info(f"View the complete evaluation report at {report_path}")
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate crime prediction models')
    parser.add_argument('--model', type=str, help='Path to the trained model file')
    parser.add_argument('--data', type=str, help='Path to the test data file')
    parser.add_argument('--target', type=str, default='target_24h', help='Target column name')
    parser.add_argument('--quick', action='store_true', help='Evaluate the quick model')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    
    args = parser.parse_args()
    
    logger.info(f"Model evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.quick:
            logger.info("Evaluating quick model...")
            evaluate_quick_model(args.model, args.data, args.target, args.output_dir)
        else:
            # Load model
            model_data = load_model(args.model)
            if model_data is None:
                exit(1)
            
            # Load test data
            test_data = load_test_data(args.data)
            if test_data is None:
                exit(1)
            
            # Evaluate model
            evaluation_results = evaluate_model(model_data, test_data, args.target, args.output_dir)
            
            # Create a report
            if evaluation_results:
                create_evaluation_report(evaluation_results, args.model, args.data)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during evaluation: {str(e)}")
    
    logger.info(f"Process finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



