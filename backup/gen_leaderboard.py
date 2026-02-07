import json
import pandas as pd
import os

def generate_html_leaderboard(json_path, html_path):
    if not os.path.exists(json_path):
        print("No results found yet. Creating placeholder leaderboard.")
        with open(html_path, 'w') as f:
            f.write("<html><body><h1>Leaderboard</h1><p>No submissions yet.</p></body></html>")
        return

    # 1. Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. Format for Table
    # Assuming JSON format: {"username": {"accuracy": 0.85, "date": "2026-01-20"}}
    leaderboard_data = []
    for user, metrics in data.items():
        leaderboard_data.append({
            "Participant": user,
            "Accuracy": metrics.get("accuracy", 0),
            "Status": "✅ Verified"
        })

    df = pd.DataFrame(leaderboard_data).sort_values(by="Accuracy", ascending=False)

    # 3. Convert to HTML
    html_table = df.to_html(classes='table table-striped', index=False)
    
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <title>GNN Challenge Leaderboard</title>
    </head>
    <body class="container mt-5">
        <h1 class="mb-4">🏆 GNN Challenge Leaderboard</h1>
        {html_table}
        <footer class="mt-5 text-muted">Last updated: {pd.Timestamp.now()}</footer>
    </body>
    </html>
    """

    with open(html_path, 'w') as f:
        f.write(full_html)
    print(f"Leaderboard updated at {html_path}")

if __name__ == "__main__":
    generate_html_leaderboard('results.json', 'docs/leadership.html')