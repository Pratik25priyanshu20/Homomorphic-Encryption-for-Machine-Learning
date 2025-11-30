"""
Logistic Regression Model for Heart Disease Prediction

This model serves as:
1. Baseline for plaintext ML performance
2. Foundation for encrypted inference (Day 3-4)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
from typing import Dict, Tuple
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time


class LogisticRegressionModel:
    """
    Logistic Regression classifier for heart disease prediction
    
    Why Logistic Regression?
        - Simple, interpretable
        - Fast inference (<1ms)
        - Easy to convert to encrypted version (linear operations)
        - Good baseline performance
    
    Model equation: 
        y = sigmoid(w1*x1 + w2*x2 + ... + wn*xn + b)
        
    This is fully polynomial-friendly for homomorphic encryption!
    """
    
    def __init__(self, penalty: str = 'l2', C: float = 1.0, max_iter: int = 1000):
        """
        Initialize Logistic Regression model
        
        Args:
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength (smaller = stronger)
            max_iter: Maximum iterations for convergence
        """
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            random_state=42,
            solver='lbfgs'
        )
        self.is_trained = False
        self.training_time = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict:
        """
        Train logistic regression model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            verbose: Print training info
            
        Returns:
            Training metrics dictionary
        """
        if verbose:
            print("\n" + "="*60)
            print("🚀 TRAINING LOGISTIC REGRESSION")
            print("="*60)
            print(f"Training samples: {len(X_train)}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Positive samples: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        if verbose:
            print(f"\n⏱️  Training completed in {self.training_time:.3f} seconds")
            
            # Model parameters
            print(f"\n📊 Model Parameters:")
            print(f"   Coefficients shape: {self.model.coef_.shape}")
            print(f"   Intercept: {self.model.intercept_[0]:.4f}")
            print(f"   Number of iterations: {self.model.n_iter_[0]}")
        
        # Training accuracy
        train_acc = self.model.score(X_train, y_train)
        
        metrics = {
            'training_time': self.training_time,
            'train_accuracy': train_acc,
            'n_features': X_train.shape[1],
            'n_samples': len(X_train)
        }
        
        if verbose:
            print(f"\n✅ Training Accuracy: {train_acc*100:.2f}%")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class
        
        Args:
            X: Input features
            
        Returns:
            Probability of class 1 (disease present)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        # Predictions
        start_time = time.time()
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'inference_time_ms': inference_time
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        if verbose:
            print("\n" + "="*60)
            print("📊 EVALUATION RESULTS - LOGISTIC REGRESSION")
            print("="*60)
            print(f"\n🎯 Classification Metrics:")
            print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"   Precision: {metrics['precision']*100:.2f}%")
            print(f"   Recall:    {metrics['recall']*100:.2f}%")
            print(f"   F1-Score:  {metrics['f1']*100:.2f}%")
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            print(f"\n⚡ Performance:")
            print(f"   Inference time: {metrics['inference_time_ms']:.3f} ms/sample")
            
            print(f"\n🔍 Confusion Matrix:")
            print(f"   True Negatives:  {cm[0,0]:3d}  |  False Positives: {cm[0,1]:3d}")
            print(f"   False Negatives: {cm[1,0]:3d}  |  True Positives:  {cm[1,1]:3d}")
            
            # Classification report
            print(f"\n📋 Detailed Report:")
            print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
            
            print("="*60)
        
        return metrics
    
    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Optional path to save figure
        """
        y_proba = self.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curve saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix heatmap
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Optional path to save figure
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Logistic Regression')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, feature_names: list = None) -> Dict:
        """
        Get feature importance (model coefficients)
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        coef = self.model.coef_[0]
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(coef))]
        
        importance = dict(zip(feature_names, coef))
        
        # Sort by absolute value
        importance_sorted = dict(sorted(importance.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True))
        
        return importance_sorted
    
    def save(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath: Path to save model (e.g., 'models/plaintext/logistic_regression.pkl')
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'training_time': self.training_time
        }, filepath)
        
        print(f"💾 Model saved to: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load trained model
        
        Args:
            filepath: Path to saved model
        """
        state = joblib.load(filepath)
        
        self.model = state['model']
        self.is_trained = state['is_trained']
        self.training_time = state.get('training_time')
        
        print(f"📂 Model loaded from: {filepath}")


# Example usage
if __name__ == '__main__':
    from src.data.preprocessor import HeartDiseasePreprocessor
    
    # Load and prepare data
    preprocessor = HeartDiseasePreprocessor()
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Train model
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    
    # Evaluate
    metrics = lr_model.evaluate(X_test, y_test)
    
    # Save model
    lr_model.save('../../models/plaintext/logistic_regression.pkl')
    
    print("\n🎉 Logistic Regression training complete!")