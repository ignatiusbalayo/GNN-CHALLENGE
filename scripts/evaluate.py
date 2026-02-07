"""
Updated evaluate.py - Works with anonymized labels (Class_A, Class_B, etc.)
"""

import pandas as pd
import json
import os
from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(submission_path, target_path):
    """
    Calculate accuracy and Macro-F1 for anonymized labels
    """
    # Check if files exist
    if not os.path.exists(submission_path) or os.path.getsize(submission_path) == 0:
        print(f"Error: Submission file {submission_path} is missing or empty.")
        return None
    
    if not os.path.exists(target_path):
        print(f"Error: Target labels file {target_path} not found.")
        return None

    try:
        sub = pd.read_csv(submission_path)
        true = pd.read_csv(target_path)
        
        # Check for correct column names
        # Support both 'label' and 'pred_label'
        pred_col = 'pred_label' if 'pred_label' in sub.columns else 'label'
        
        if 'id' not in sub.columns or pred_col not in sub.columns:
            print(f"Error: Submission must contain 'id' and '{pred_col}' columns.")
            return None
        
        if 'id' not in true.columns or 'label' not in true.columns:
            print(f"Error: Target labels must contain 'id' and 'label' columns.")
            return None

        # Merge on ID to ensure correct alignment
        merged = sub.merge(true, on='id', suffixes=('_pred', '_true'))
        
        if len(merged) != len(true):
            print(f"Error: ID mismatch. Expected {len(true)} IDs, got {len(merged)} matches.")
            return None
        
        # Get predictions and true labels
        y_pred = merged[pred_col].values
        y_true = merged['label'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred) * 100  # Convert to percentage
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class F1 (optional)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0,
                                labels=['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 'Class_F'])
        
        return {
            "accuracy": round(float(accuracy), 2),
            "macro_f1": round(float(macro_f1), 4),
            "per_class_f1": {
                'Class_A': round(float(per_class_f1[0]), 4),
                'Class_B': round(float(per_class_f1[1]), 4),
                'Class_C': round(float(per_class_f1[2]), 4),
                'Class_D': round(float(per_class_f1[3]), 4),
                'Class_E': round(float(per_class_f1[4]), 4),
                'Class_F': round(float(per_class_f1[5]), 4),
            }
        }
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Paths
    latest_sub = "submissions/submission.csv" 
    target_labels = "data/private_labels.csv"

    scores = calculate_metrics(latest_sub, target_labels)
    
    if scores:
        # Load existing results
        results = {}
        if os.path.exists("results.json"):
            with open("results.json", "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}
        
        # Get username
        user = os.getenv('GITHUB_ACTOR', 'local_user')
        
        # Add/update user's score
        results[user] = scores
        
        # Save results
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Evaluation successful!")
        print(f"   User: {user}")
        print(f"   Accuracy: {scores['accuracy']}%")
        print(f"   Macro-F1: {scores['macro_f1']}")
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()