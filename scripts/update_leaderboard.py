"""
Leaderboard Updater Script
Updates the leaderboard HTML with new submission results
Merged version: Supports nested metrics and 425-node evaluation
"""

import json
import sys
from datetime import datetime
import os

def update_leaderboard(results_file):
    """
    Update leaderboard with new submission results
    """
    # 1. Load results with error handling
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return

    # CHECK: Use .get() to avoid KeyError if 'status' is missing
    if results.get('status') != 'success':
        print(f"Skipping leaderboard update - submission status: {results.get('status')}")
        return
    
    # 2. Extract metrics (Handling the nested dictionary)
    # This is where the KeyError 'accuracy' was coming from
    metrics = results.get('metrics', {})
    
    username = results.get('username', 'unknown')
    accuracy = metrics.get('accuracy', 0)
    macro_f1 = metrics.get('macro_f1', 0)

    # Load existing leaderboard data
    leaderboard_file = 'docs/leaderboard_data.json'
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            leaderboard_data = json.load(f)
    else:
        leaderboard_data = []
    
    # 3. Create new entry
    new_entry = {
        'username': username,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        # Get per_class_f1 from metrics or top level depending on grader version
        'per_class_f1': metrics.get('per_class_f1', results.get('per_class_f1', {}))
    }
    
    # Check if user already has submission
    existing_idx = None
    for i, entry in enumerate(leaderboard_data):
        if entry['username'] == username:
            existing_idx = i
            break
    
    # Update or append
    if existing_idx is not None:
        if new_entry['macro_f1'] > leaderboard_data[existing_idx].get('macro_f1', 0):
            leaderboard_data[existing_idx] = new_entry
            print(f"Updated {username}'s score (improved!)")
        else:
            print(f"Kept {username}'s previous better score")
    else:
        leaderboard_data.append(new_entry)
        print(f"Added new entry for {username}")
    
    # Sort and rank
    leaderboard_data.sort(key=lambda x: x.get('macro_f1', 0), reverse=True)
    for i, entry in enumerate(leaderboard_data):
        entry['rank'] = i + 1
    
    # Save JSON data
    os.makedirs('docs', exist_ok=True)
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard_data, f, indent=2)
    
    # 4. Generate HTML
    generate_leaderboard_html(leaderboard_data)
    print(f"✅ Leaderboard updated for {username}!")

# ... Keep your generate_leaderboard_html function exactly as it is ...
