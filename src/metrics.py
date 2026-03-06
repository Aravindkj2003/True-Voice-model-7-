"""
Evaluation metrics for Deepfake Audio Detection System
Including EER (Equal Error Rate) calculation
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    """Calculate various evaluation metrics including EER"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.scores = []
        self.targets = []
    
    def add_batch(self, scores, targets):
        """
        Add batch of predictions and targets
        
        Args:
            scores: Model output scores (before argmax) shape: (batch_size, 2)
            targets: Ground truth labels shape: (batch_size,)
        """
        self.scores.append(scores.detach().cpu().numpy())
        self.predictions.append(np.argmax(scores.detach().cpu().numpy(), axis=1))
        self.targets.append(targets.detach().cpu().numpy())
    
    def get_metrics(self):
        """Calculate all metrics"""
        if len(self.predictions) == 0:
            return {}
        
        # Concatenate all batches
        predictions = np.concatenate(self.predictions, axis=0)
        scores = np.concatenate(self.scores, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        
        # Get softmax probabilities for fake class (class 1)
        softmax_scores = self._softmax(scores)
        fake_probabilities = softmax_scores[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1': f1_score(targets, predictions, zero_division=0),
            'eer': self.calculate_eer(targets, fake_probabilities),
            'auc': self.calculate_auc(targets, fake_probabilities),
        }
        
        # Add confusion matrix details
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        metrics['far'] = float(fp) / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Rate
        metrics['miss'] = float(fn) / (fn + tp) if (fn + tp) > 0 else 0  # Miss Rate
        
        return metrics
    
    @staticmethod
    def _softmax(x):
        """Calculate softmax from logits"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    @staticmethod
    def calculate_eer(targets, scores):
        """
        Calculate Equal Error Rate (EER)
        EER is the point where False Alarm Rate equals Miss Rate
        
        Args:
            targets: Ground truth labels (0=Real, 1=Fake)
            scores: Probability scores for fake class
            
        Returns:
            eer: Equal Error Rate value
            eer_threshold: Threshold at which EER occurs
        """
        # Convert targets to binary (0=Real, 1=Fake)
        fpr, fnr, thresholds = roc_curve(targets, scores)
        fnr = 1.0 - fnr  # Convert TPR to FNR
        
        # Find the threshold where FPR == FNR
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = max(fpr[min_index], fnr[min_index])
        eer_threshold = thresholds[min_index]
        
        return float(eer)
    
    @staticmethod
    def calculate_auc(targets, scores):
        """
        Calculate Area Under the ROC Curve (AUC)
        
        Args:
            targets: Ground truth labels
            scores: Probability scores
            
        Returns:
            auc: AUC score
        """
        fpr, tpr, _ = roc_curve(targets, scores)
        return float(auc(fpr, tpr))


def get_metrics_dict(scores, targets):
    """Utility function to get metrics from scores and targets"""
    calculator = MetricsCalculator()
    calculator.add_batch(scores, targets)
    return calculator.get_metrics()
