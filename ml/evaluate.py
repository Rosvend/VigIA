"""
Comprehensive Model Evaluation Script for Crime Prediction
This script provides detailed evaluation of trained crime prediction models
with focus on ROC AUC, PR AUC, log loss, and precision metrics.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
import argparse
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrimePredictionEvaluator:
    """Comprehensive evaluator for crime prediction models"""
    
    def __init__(self, model_path=None, dataset_path=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model file
            dataset_path: Path to temporal dataset for evaluation
        """
        self.model_data = None
        self.dataset = None
        self.results = {}
        
        if model_path:
            self.load_model(model_path)
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model_data = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            if 'metrics' in self.model_data:
                logger.info("Model training metrics:")
                for split, metrics in self.model_data['metrics'].items():
                    logger.info(f"  {split.title()}: ROC AUC={metrics.get('roc_auc', 0):.4f}, "
                              f"PR AUC={metrics.get('pr_auc', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_dataset(self, dataset_path):
        """Load temporal dataset"""
        try:
            if dataset_path.endswith('.pkl'):
                self.dataset = pd.read_pickle(dataset_path)
            elif dataset_path.endswith('.csv'):
                self.dataset = pd.read_csv(dataset_path)
            else:
                raise ValueError("Dataset must be .pkl or .csv file")
            
            logger.info(f"Loaded dataset with {len(self.dataset)} samples from {dataset_path}")
            
            # Check for target columns
            target_cols = [col for col in self.dataset.columns if col.startswith('target_')]
            logger.info(f"Found target columns: {target_cols}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def create_temporal_splits(self, target_col='target_12h'):
        """Create temporal train/validation/test splits"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        # Ensure reference_time is datetime
        self.dataset['reference_time'] = pd.to_datetime(self.dataset['reference_time'])
        self.dataset['ref_month'] = self.dataset['reference_time'].dt.month
        
        # Create splits: Jan-Sep (train), Oct-Nov (val), Dec (test)
        train_data = self.dataset[self.dataset['ref_month'] <= 9].copy()
        val_data = self.dataset[
            (self.dataset['ref_month'] > 9) & 
            (self.dataset['ref_month'] <= 11)
        ].copy()
        test_data = self.dataset[self.dataset['ref_month'] > 11].copy()
        
        logger.info(f"Temporal splits created:")
        logger.info(f"  Training: {len(train_data)} samples (Jan-Sep)")
        logger.info(f"  Validation: {len(val_data)} samples (Oct-Nov)")
        logger.info(f"  Test: {len(test_data)} samples (Dec)")
        
        # Check class distributions
        for name, data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
            if target_col in data.columns:
                pos_count = data[target_col].sum()
                total = len(data)
                logger.info(f"  {name} - Positive class: {pos_count}/{total} ({pos_count/total:.2%})")
        
        return train_data, val_data, test_data
    
    def prepare_features(self, data, target_col):
        """Prepare features for model evaluation"""
        # Drop unnecessary columns
        drop_cols = ['geometry', 'reference_time', 'ref_month'] if 'geometry' in data.columns else ['reference_time', 'ref_month']
        drop_cols += [col for col in data.columns if col.startswith('target_') and col != target_col]
        drop_cols += ['h3_index']  # Keep separate for analysis
        
        X = data.drop(drop_cols, axis=1, errors='ignore')
        y = data[target_col]
        
        return X, y
    
    def evaluate_comprehensive(self, X, y, split_name="Test", output_dir="evaluation_plots"):
        """
        Perform comprehensive evaluation with all required metrics
        
        Args:
            X: Features
            y: True labels
            split_name: Name of the data split
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with all metrics
        """
        if self.model_data is None:
            raise ValueError("Model not loaded")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model = self.model_data['model']
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Calculate all required metrics
        metrics = {
            'split': split_name,
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'positive_rate': float(y.mean()),
            'roc_auc': roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0,
            'pr_auc': average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0,
            'log_loss': log_loss(y, y_prob) if len(np.unique(y)) > 1 else 1.0,
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'accuracy': float((y_pred == y).mean())
        }
        
        # Print results
        logger.info(f"\n=== {split_name} Evaluation Results ===")
        logger.info(f"Dataset: {metrics['n_samples']} samples, {metrics['n_positive']} positive ({metrics['positive_rate']:.2%})")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Generate plots
        self.plot_roc_curve(y, y_prob, split_name, output_dir)
        self.plot_precision_recall_curve(y, y_prob, split_name, output_dir)
        self.plot_prediction_distribution(y, y_prob, split_name, output_dir)
        self.plot_confusion_matrix(cm, split_name, output_dir)
        
        # Store results
        self.results[split_name.lower()] = metrics
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_prob, split_name, output_dir):
        """Plot ROC curve"""
        if len(np.unique(y_true)) <= 1:
            logger.warning(f"Cannot plot ROC curve for {split_name} - only one class present")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {split_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, f'roc_curve_{split_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")
    
    def plot_precision_recall_curve(self, y_true, y_prob, split_name, output_dir):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) <= 1:
            logger.warning(f"Cannot plot PR curve for {split_name} - only one class present")
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.8, where='post', linewidth=2)
        plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {split_name}\nAP = {avg_precision:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Add baseline (random classifier)
        baseline = y_true.mean()
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Random (AP = {baseline:.3f})')
        plt.legend()
        
        output_path = os.path.join(output_dir, f'pr_curve_{split_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"PR curve saved to {output_path}")
    
    def plot_prediction_distribution(self, y_true, y_prob, split_name, output_dir):
        """Plot distribution of prediction probabilities"""
        plt.figure(figsize=(10, 6))
        
        # Separate probabilities by true class
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        
        plt.subplot(1, 2, 1)
        plt.hist([neg_probs, pos_probs], bins=30, alpha=0.7, 
                label=['No Crime (0)', 'Crime (1)'], density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Prediction Distribution - {split_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calibration plot
        plt.subplot(1, 2, 2)
        if len(np.unique(y_true)) > 1:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                    label="Model", linewidth=2)
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'prediction_dist_{split_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Prediction distribution saved to {output_path}")
    
    def plot_confusion_matrix(self, cm, split_name, output_dir):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Crime', 'Crime'],
                   yticklabels=['No Crime', 'Crime'])
        plt.title(f'Confusion Matrix - {split_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        output_path = os.path.join(output_dir, f'confusion_matrix_{split_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def compare_models(self, model_paths, dataset_path, target_col='target_12h'):
        """Compare multiple models on the same dataset"""
        logger.info(f"Comparing {len(model_paths)} models...")
        
        # Load dataset
        self.load_dataset(dataset_path)
        
        # Create temporal splits
        train_data, val_data, test_data = self.create_temporal_splits(target_col)
        X_test, y_test = self.prepare_features(test_data, target_col)
        
        comparison_results = []
        
        for i, model_path in enumerate(model_paths):
            logger.info(f"\nEvaluating model {i+1}/{len(model_paths)}: {model_path}")
            
            # Load model
            self.load_model(model_path)
            
            # Evaluate
            metrics = self.evaluate_comprehensive(
                X_test, y_test, 
                split_name=f"Model_{i+1}",
                output_dir=f"evaluation_model_{i+1}"
            )
            
            metrics['model_path'] = model_path
            comparison_results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv('model_comparison_detailed.csv', index=False)
        logger.info("Model comparison saved to model_comparison_detailed.csv")
        
        # Print summary
        logger.info("\n=== MODEL COMPARISON SUMMARY ===")
        for _, row in comparison_df.iterrows():
            logger.info(f"Model: {row['model_path']}")
            logger.info(f"  ROC AUC: {row['roc_auc']:.4f}")
            logger.info(f"  PR AUC: {row['pr_auc']:.4f}")
            logger.info(f"  Log Loss: {row['log_loss']:.4f}")
            logger.info(f"  Precision: {row['precision']:.4f}")
        
        return comparison_df
    
    def generate_evaluation_report(self, target_col='target_12h', output_file='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        if not self.results:
            logger.warning("No evaluation results found. Run evaluation first.")
            return
        
        with open(output_file, 'w') as f:
            f.write("CRIME PREDICTION MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target: {target_col}\n\n")
            
            if self.model_data:
                f.write("MODEL INFORMATION:\n")
                f.write(f"  Prediction Window: {self.model_data.get('prediction_window', 'Unknown')} hours\n")
                f.write(f"  Training Date: {self.model_data.get('training_date', 'Unknown')}\n")
                f.write(f"  SMOTE Used: {self.model_data.get('use_smote', 'Unknown')}\n\n")
            
            for split_name, metrics in self.results.items():
                f.write(f"{split_name.upper()} SET RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Samples: {metrics['n_samples']}\n")
                f.write(f"  Positive samples: {metrics['n_positive']} ({metrics['positive_rate']:.2%})\n")
                f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"  PR AUC: {metrics['pr_auc']:.4f}\n")
                f.write(f"  Log Loss: {metrics['log_loss']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n\n")
        
        logger.info(f"Evaluation report saved to {output_file}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate crime prediction models')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to temporal dataset')
    parser.add_argument('--target', type=str, default='target_12h', help='Target column name')
    parser.add_argument('--compare', nargs='+', help='Multiple model paths for comparison')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        evaluator = CrimePredictionEvaluator()
        
        if args.compare:
            # Compare multiple models
            logger.info("Running model comparison...")
            comparison_df = evaluator.compare_models(args.compare, args.dataset, args.target)
            
        else:
            # Evaluate single model
            logger.info("Evaluating single model...")
            evaluator.load_model(args.model)
            evaluator.load_dataset(args.dataset)
            
            # Create temporal splits
            train_data, val_data, test_data = evaluator.create_temporal_splits(args.target)
            
            # Evaluate on each split
            for split_name, data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
                if len(data) > 0:
                    X, y = evaluator.prepare_features(data, args.target)
                    evaluator.evaluate_comprehensive(X, y, split_name, args.output_dir)
            
            # Generate report
            evaluator.generate_evaluation_report(args.target, 
                                                os.path.join(args.output_dir, 'evaluation_report.txt'))
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()



