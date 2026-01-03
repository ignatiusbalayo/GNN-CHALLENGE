import pandas as pd
import sys
import io
import os
from sklearn.metrics import accuracy_score

def grade():
    # 1. Get Secret Key from Environment
    secret_content = os.environ.get("SECRET_KEY")
    if not secret_content:
        print("ERROR: Secret Key is missing in GitHub Settings.")
        return

    try:
        # 2. Read Files
        truth = pd.read_csv(io.StringIO(secret_content))
        student_file = sys.argv[1]
        pred = pd.read_csv(student_file)
        
        # 3. Merge & Score
        # We merge on 'id' to ensure we compare the right nodes
        merged = pd.merge(pred, truth, on='id')
        acc = accuracy_score(merged['label'], merged['pred_label'])
        
        # 4. Print Score (The Robot reads this)
        print(f"ACCURACY:{acc*100:.2f}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    grade()