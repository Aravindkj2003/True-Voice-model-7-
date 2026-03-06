"""
Evaluation script for testing trained model
"""

import torch
import argparse
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import DATASET_CONFIG, TRAINING_CONFIG, MODEL_CONFIG, PATHS
from src.model import get_model
from src.data_loader import DataManager
from src.metrics import MetricsCalculator
from src.utils import get_device, set_seed
from tqdm import tqdm


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device to use
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            metrics_calc.add_batch(outputs, targets)
    
    metrics = metrics_calc.get_metrics()
    return metrics


def print_evaluation_results(metrics):
    """Print evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"AUC:          {metrics['auc']:.4f}")
    print(f"EER:          {metrics['eer']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  True Positives:  {metrics['tp']}")
    
    print(f"\nError Rates:")
    print(f"  False Alarm Rate (FAR): {metrics['far']:.4f}")
    print(f"  Miss Rate (FNR):        {metrics['miss']:.4f}")
    
    print("="*80 + "\n")


def main(args):
    """Main evaluation function"""
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Create model
    model = get_model(
        num_classes=MODEL_CONFIG['num_classes'],
        backbone=MODEL_CONFIG['backbone'],
        pretrained=False,
        device=device,
        dropout_rate=MODEL_CONFIG['dropout_rate']
    )
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully\n")
    
    # Load test data
    data_manager = DataManager(
        data_dir=PATHS['data_dir'],
        sample_rate=DATASET_CONFIG['sample_rate'],
        batch_size=TRAINING_CONFIG['batch_size'],
        num_workers=TRAINING_CONFIG['num_workers'],
    )
    
    print("Loading test dataset...")
    test_loader = data_manager.load_from_directory_structure(split='test')
    
    # Evaluate
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print_evaluation_results(metrics)
    
    # Save results
    if args.save_results:
        results_path = os.path.join(PATHS['result_dir'], 'evaluation_results.json')
        os.makedirs(PATHS['result_dir'], exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Deepfake Detection Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save evaluation results to JSON')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    main(args)
