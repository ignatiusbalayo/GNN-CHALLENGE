import pandas as pd
import json
import os

def calculate_metrics(submission_path, target_path):
    # Check if the submission file exists and is not empty
    if not os.path.exists(submission_path) or os.path.getsize(submission_path) == 0:
        print(f"Error: Submission file {submission_path} is missing or empty.")
        return None
    
    # Check if the private labels file exists
    if not os.path.exists(target_path):
        print(f"Error: Target labels file {target_path} not found.")
        return None

    try:
        sub = pd.read_csv(submission_path)
        true = pd.read_csv(target_path)
        
        # Ensure 'id' and 'label' columns exist
        if 'id' not in sub.columns or 'label' not in sub.columns:
            print("Error: Submission must contain 'id' and 'label' columns.")
            return None

        # Sort to ensure matching order
        true = true.sort_values('id')
        sub = sub.sort_values('id')
        
        accuracy = (sub['label'].values == true['label'].values).mean()
        return {"accuracy": round(float(accuracy), 4)}
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return None

def main():
    # Use paths relative to the project root
    latest_sub = "submissions/submission.csv" 
    target_labels = "data/private_labels.csv" # Ensure this file was created by setup!

    scores = calculate_metrics(latest_sub, target_labels)
    
    if scores:
        results = {}
        if os.path.exists("results.json"):
            with open("results.json", "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = {}
        
        user = os.getenv('GITHUB_ACTOR', 'local_user')
        results[user] = scores
        
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation successful. Score: {scores}")

if __name__ == "__main__":
    main()