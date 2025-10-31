"""
Metrics evaluation and plotting utilities
Calculates precision, recall, F1-score, confusion matrix, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

from config import ProjectConfig


class MetricsCalculator:
    """Calculate and visualize classification metrics"""
    
    def __init__(self, config: ProjectConfig = None):
        self.config = config or ProjectConfig()
        self.labels = self.config.labels
        
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict:
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities (optional, for ROC)
            average: Averaging method for multi-class ('weighted', 'macro', 'micro')
        
        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
        )
        
        # Accuracy
        accuracy = (y_true == y_pred).sum() / len(y_true)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.labels)))
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'labels': self.labels,
        }
        
        # ROC curve for binary or per-class
        if y_probs is not None:
            roc_metrics = self._calculate_roc(y_true, y_probs)
            metrics.update(roc_metrics)
        
        return metrics
    
    def _calculate_roc(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict:
        """Calculate ROC curve metrics"""
        roc_metrics = {}
        
        if len(self.labels) == 2:
            # Binary classification
            fpr, tpr, thresholds = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_metrics['roc_auc'] = float(roc_auc)
            roc_metrics['roc_fpr'] = fpr.tolist()
            roc_metrics['roc_tpr'] = tpr.tolist()
            roc_metrics['roc_thresholds'] = thresholds.tolist()
        else:
            # Multi-class: one-vs-rest for each class
            roc_aucs = []
            for i, label in enumerate(self.labels):
                # One-vs-rest
                y_binary = (y_true == i).astype(int)
                if y_binary.sum() > 0:  # Make sure class exists
                    fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_aucs.append(roc_auc)
                    roc_metrics[f'roc_auc_{label}'] = float(roc_auc)
                    roc_metrics[f'roc_fpr_{label}'] = fpr.tolist()
                    roc_metrics[f'roc_tpr_{label}'] = tpr.tolist()
            
            if roc_aucs:
                roc_metrics['roc_auc_macro'] = float(np.mean(roc_aucs))
        
        return roc_metrics
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot confusion matrix"""
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {save_path}")
        else:
            plt.show()
    
    def plot_metrics_bar(
        self,
        metrics: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Plot precision, recall, F1 per class"""
        precision_per_class = metrics['precision_per_class']
        recall_per_class = metrics['recall_per_class']
        f1_per_class = metrics['f1_per_class']
        
        x = np.arange(len(self.labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#4CAF50')
        bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#2196F3')
        bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='#FF9800')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics per Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics bar chart to: {save_path}")
        else:
            plt.show()
    
    def plot_roc_curve(
        self,
        metrics: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot ROC curve(s)"""
        plt.figure(figsize=figsize)
        
        if len(self.labels) == 2:
            # Binary
            fpr = metrics['roc_fpr']
            tpr = metrics['roc_tpr']
            auc_score = metrics['roc_auc']
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {auc_score:.3f})')
        else:
            # Multi-class: plot each class
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))
            for i, label in enumerate(self.labels):
                fpr_key = f'roc_fpr_{label}'
                tpr_key = f'roc_tpr_{label}'
                auc_key = f'roc_auc_{label}'
                if fpr_key in metrics:
                    fpr = metrics[fpr_key]
                    tpr = metrics[tpr_key]
                    auc_score = metrics[auc_key]
                    plt.plot(fpr, tpr, color=colors[i], lw=2,
                           label=f'{label} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve to: {save_path}")
        else:
            plt.show()
    
    def plot_all_metrics(
        self,
        metrics: Dict,
        save_dir: Optional[str] = None,
        prefix: str = "metrics"
    ):
        """Plot all metrics and save to directory"""
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        cm_path = f"{save_dir}/{prefix}_confusion_matrix.png" if save_dir else None
        self.plot_confusion_matrix(cm, save_path=cm_path)
        plt.close()
        
        # Metrics bar chart
        bar_path = f"{save_dir}/{prefix}_metrics_bar.png" if save_dir else None
        self.plot_metrics_bar(metrics, save_path=bar_path)
        plt.close()
        
        # ROC curve (if available)
        if 'roc_auc' in metrics or any(f'roc_auc_{label}' in metrics for label in self.labels):
            roc_path = f"{save_dir}/{prefix}_roc_curve.png" if save_dir else None
            self.plot_roc_curve(metrics, save_path=roc_path)
            plt.close()
        
        # Save metrics as JSON
        if save_dir:
            json_path = Path(save_dir) / f"{prefix}_metrics.json"
            # Remove numpy arrays from dict for JSON serialization
            json_metrics = {k: v for k, v in metrics.items() 
                          if not isinstance(v, (list, np.ndarray)) or k == 'labels'}
            json_metrics.update({
                'confusion_matrix': metrics['confusion_matrix'],
                'precision_per_class': metrics['precision_per_class'],
                'recall_per_class': metrics['recall_per_class'],
                'f1_per_class': metrics['f1_per_class'],
            })
            
            with open(json_path, 'w') as f:
                json.dump(json_metrics, f, indent=2)
            print(f"Saved metrics JSON to: {json_path}")
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("Classification Report")
        print("="*60)
        print(classification_report(
            y_true, y_pred, target_names=self.labels, zero_division=0
        ))
        print("="*60)


def evaluate_model(
    model,
    dataloader,
    device,
    config: ProjectConfig = None,
    save_plots: bool = True,
    plot_dir: str = "plots"
) -> Dict:
    """
    Evaluate model and calculate all metrics
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        config: Project configuration
        save_plots: Whether to save plots
        plot_dir: Directory to save plots
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            # Handle different input shapes
            if features.dim() == 2:
                features = features.unsqueeze(1)
            elif features.dim() == 3 and features.shape[1] != 1:
                features = features.unsqueeze(1)
            
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    calc = MetricsCalculator(config)
    metrics = calc.calculate_metrics(y_true, y_pred, y_probs)
    
    # Print report
    calc.print_classification_report(y_true, y_pred)
    
    # Print summary
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall: {metrics['recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    elif 'roc_auc_macro' in metrics:
        print(f"Macro Average ROC AUC: {metrics['roc_auc_macro']:.4f}")
    
    # Plot metrics
    if save_plots:
        calc.plot_all_metrics(metrics, save_dir=plot_dir, prefix="evaluation")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Metrics module loaded successfully!")
    print("Use evaluate_model() to evaluate a trained model")
    print("Or use MetricsCalculator directly for custom evaluation")

