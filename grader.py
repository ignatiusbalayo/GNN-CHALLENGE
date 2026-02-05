"""
Grader Script for GNN Challenge
Evaluates submission CSV files against private test labels
"""

import pandas as pd
import numpy as np
import sys
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os

def validate_submission(submission_path, test_ids_path, private_labels_path):
    """
    Validate and grade a submission
    
    Returns:
        dict with results or error
    """
    
    try:
        # Load submission
        submission = pd.read_csv(submission_path)
        
        # Load expected test IDs
        test_data = pd.read_csv(test_ids_path)
        expected_ids = set(test_data['id'].values)
        
        # Load private labels
        private_labels = pd.read_csv(private_labels_path)
        
        # Validation 1: Check columns
        if 'id' not in submission.columns or 'pred_label' not in submission.columns:
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Missing required columns. Found: {list(submission.columns)}. Expected: ['id', 'pred_label']"
            }
        
        # Validation 2: Check for extra columns
        if len(submission.columns) != 2:
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Too many columns. Expected exactly 2 columns (id, pred_label), found {len(submission.columns)}"
            }
        
        # Validation 3: Check number of predictions
        if len(submission) != len(expected_ids):
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Wrong number of predictions. Expected {len(expected_ids)}, got {len(submission)}"
            }
        
        # Validation 4: Check for missing IDs
        submitted_ids = set(submission['id'].values)
        missing_ids = expected_ids - submitted_ids
        if missing_ids:
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Missing {len(missing_ids)} test IDs. Example missing IDs: {list(missing_ids)[:5]}"
            }
        
        # Validation 5: Check for extra IDs
        extra_ids = submitted_ids - expected_ids
        if extra_ids:
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Found {len(extra_ids)} unexpected IDs. Example: {list(extra_ids)[:5]}"
            }
        
        # Validation 6: Check for duplicates
        if submission['id'].duplicated().any():
            duplicates = submission[submission['id'].duplicated()]['id'].values
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Duplicate IDs found: {duplicates[:5]}"
            }
        
        # Validation 7: Check valid labels
        valid_labels = {'Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 'Class_F'}
        invalid_labels = set(submission['pred_label'].unique()) - valid_labels
        if invalid_labels:
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Invalid labels found: {invalid_labels}. Must be one of: {valid_labels}"
            }
        
        # Merge submission with private labels
        merged = submission.merge(private_labels, on='id', how='inner')
        
        if len(merged) != len(expected_ids):
            return {
                'status': 'error',
                'username': os.path.basename(submission_path).replace('.csv', ''),
                'accuracy': 0,
                'macro_f1': 0,
                'message': f"Merging error. This shouldn't happen - contact organizers."
            }
        
        # Calculate metrics
        y_true = merged['label'].values
        y_pred = merged['pred_label'].values
        
        accuracy = accuracy_score(y_true, y_pred) * 100
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class F1 scores
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, 
                                labels=['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 'Class_F'])
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, 
                                       labels=['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 'Class_F'])
        
        # Extract username from file path
        username = os.path.basename(submission_path).replace('.csv', '')
        
        return {
            'status': 'success',
            'username': username,
            'accuracy': round(accuracy, 2),
            'macro_f1': round(macro_f1, 4),
            'per_class_f1': {
                'Class_A': round(per_class_f1[0], 4),
                'Class_B': round(per_class_f1[1], 4),
                'Class_C': round(per_class_f1[2], 4),
                'Class_D': round(per_class_f1[3], 4),
                'Class_E': round(per_class_f1[4], 4),
                'Class_F': round(per_class_f1[5], 4),
            },
            'confusion_matrix': conf_matrix.tolist(),
            'message': 'Submission graded successfully!'
        }
        
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'username': os.path.basename(submission_path).replace('.csv', '') if submission_path else 'unknown',
            'accuracy': 0,
            'macro_f1': 0,
            'message': f"File not found: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'username': os.path.basename(submission_path).replace('.csv', '') if submission_path else 'unknown',
            'accuracy': 0,
            'macro_f1': 0,
            'message': f"Unexpected error: {str(e)}"
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        result = {
            'status': 'error',
            'username': 'unknown',
            'accuracy': 0,
            'macro_f1': 0,
            'message': 'Usage: python grader.py <submission_file>'
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)
    
    submission_path = sys.argv[1]
    test_ids_path = 'data/test_no_label.csv'
    private_labels_path = 'data/private_labels.csv'
    
    result = validate_submission(submission_path, test_ids_path, private_labels_path)
    
    # Print result as JSON
    print(json.dumps(result, indent=2))
    
    # Exit with error code if validation failed
    if result['status'] == 'error':
        sys.exit(1)
    else:
        sys.exit(0)