import dgl
import pandas as pd
import os
from dgl.data import CiteseerGraphDataset
from sklearn.model_selection import train_test_split

def setup():
    print("🚀 Downloading & Processing CiteSeer Data...")
    dataset = CiteseerGraphDataset(verbose=False)
    g = dataset[0]
    
    features = g.ndata['feat'].numpy()
    labels = g.ndata['label'].numpy()
    src, dst = g.edges()

    # Create IDs for each node
    ids = list(range(g.num_nodes()))

    # --- STRATIFIED SPLIT (Fixes Class Imbalance) ---
    # 1. Split into Train (60%) and Temporary (40%)
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        ids, labels, train_size=0.6, stratify=labels, random_state=42
    )

    # 2. Split Temporary into Val (20%) and Test (20%)
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, train_size=0.5, stratify=y_temp, random_state=42
    )

    os.makedirs('data', exist_ok=True)
    
    def save_csv(indices, name, current_labels, has_label=True):
        df = pd.DataFrame(features[indices])
        df.insert(0, 'id', indices)
        if has_label:
            df.insert(1, 'label', current_labels)
        df.to_csv(f'data/{name}', index=False)
        print(f"   Saved data/{name} (Size: {len(indices)})")

    save_csv(train_ids, 'train.csv', y_train, True)
    save_csv(val_ids, 'val.csv', y_val, True)
    save_csv(test_ids, 'test_no_label.csv', None, False)
    
    pd.DataFrame({'source': src.numpy(), 'target': dst.numpy()}).to_csv('data/edges.csv', index=False)

    # Save Private Labels for the Grader
    pd.DataFrame({'id': test_ids, 'label': y_test}).to_csv('SECRET_KEY.txt', index=False)
    print("\n✅ Setup Complete! Use 'SECRET_KEY.txt' as your GitHub Repo Secret.")

if __name__ == "__main__":
    setup()
