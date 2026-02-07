"""
Final Grader Logic for NetClass Arena
Ensures unique ID matching and prevents 'merge explosion'.
"""

import pandas as pd
import numpy as np
import sys
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os

def validate_and_grade(submission_path, test_ids_path, private_labels_path):
    try:
        # 1. Load Data
        if not os.path.exists(submission_path):
            return {'status': 'error', 'message': 'Submission file not found.'}
        
        # Load files and IMMEDIATELY drop duplicates to prevent "Merge Explosion"
        submission = pd.read_csv(submission_path).drop_duplicates(subset=['id'])
        truth = pd.read_csv(private_labels_path).drop_duplicates(subset=['id'])
        test_data = pd.read_csv(test_ids_path)
        
        expected_ids = set(test_data['id'].values)
        username = os.path.basename(submission_path).replace('.csv', '')

        # 2. Structural Validation
        # Check Columns
        if not {'id', 'pred_label'}.issubset(set(submission.columns)):
            return {'status': 'error', 'message': "Missing columns ['id', 'pred_label']"}
        
        # Check Column Count (Prevent index columns)
        if len(submission.columns) != 2:
            return {'status': 'error', 'message': "Submission must have exactly 2 columns."}

        # Check Row Count against Truth (746)
        if len(submission) != len(expected_ids):
             return {'status': 'error', 'message': f"Expected {len(expected_ids)} unique predictions, got {len(submission)}"}

        # 3. Grading Logic
        # Inner join ensures we only grade nodes that exist in our answer key
        merged = pd.merge(truth, submission, on='id', how='inner')
        
        if len(merged) == 0:
            return {'status': 'error', 'message': "No matching IDs found between submission and truth."}

        y_true = merged['label'].values
        y_pred = merged['pred_label'].values
        
        # Labels for CiteSeer
        labels_order = ['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 'Class_F']

        # 4. Calculate Metrics
        acc = accuracy_score(y_true, y_pred) * 100
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # This confirms we are grading the correct subset
        samples_graded = len(merged)

        return {
            'status': 'success',
            'username': username,
            'metrics': {
                'accuracy': round(acc, 2),
                'macro_f1': round(f1_macro, 4),
                'graded_samples': samples_graded
            },
            'message': 'Graded successfully.'
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({'status': 'error', 'message': 'No submission file.'}))
        sys.exit(1)

    # Standard paths for your repo
    SUB_PATH = sys.argv[1]
    TEST_IDS = 'data/test_no_label.csv'
    TRUTH_LABELS = 'data/private_labels.csv'

    result = validate_and_grade(SUB_PATH, TEST_IDS, TRUTH_LABELS)
    print(json.dumps(result, indent=2))
    
    if result['status'] == 'error':
        sys.exit(1)
