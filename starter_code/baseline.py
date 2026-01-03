import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dgl.nn import GraphConv
import os

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading Data...")
# Handle path (whether running from root or starter_code)
data_path = 'data' if os.path.exists('data') else '../data'

try:
    train = pd.read_csv(f'{data_path}/train.csv')
    val = pd.read_csv(f'{data_path}/val.csv')
    test = pd.read_csv(f'{data_path}/test_no_label.csv')
    edges = pd.read_csv(f'{data_path}/edges.csv')
except FileNotFoundError:
    print("❌ Error: Data not found. Make sure you are in the project root!")
    exit()

# Reconstruct Graph
# We concat all nodes to create a unified feature matrix
all_nodes = pd.concat([train, val, test], sort=False).sort_values('id')
feat_cols = [c for c in all_nodes.columns if c not in ['id', 'label']]
features = torch.tensor(all_nodes[feat_cols].values, dtype=torch.float32)

g = dgl.graph((edges['source'].values, edges['target'].values))
g = dgl.add_self_loop(g)
g.ndata['feat'] = features

# Create Labels & Masks
labels = torch.full((g.num_nodes(),), -1, dtype=torch.long)
labels[train['id'].values] = torch.tensor(train['label'].values)
labels[val['id'].values] = torch.tensor(val['label'].values)

train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
train_mask[train['id'].values] = True
val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
val_mask[val['id'].values] = True

print(f"Graph Ready: {g.num_nodes()} Nodes, {g.num_edges()} Edges.")

# ==========================================
# 2. MODEL (GCN)
# ==========================================
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        h = self.conv2(g, h)
        return h

# ==========================================
# 3. TRAIN
# ==========================================
model = GCN(features.shape[1], 16, 6) # CiteSeer has 6 classes
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training...")
for e in range(101):
    logits = model(g, features)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if e % 20 == 0:
        acc = (logits.argmax(1)[val_mask] == labels[val_mask]).float().mean()
        print(f"Epoch {e} | Val Acc: {acc:.4f}")

# ==========================================
# 4. SUBMISSION
# ==========================================
print("Generating Submission...")
logits = model(g, features)
preds = logits.argmax(1).numpy()

# Extract only Test IDs
test_ids = test['id'].values
sub = pd.DataFrame({'id': test_ids, 'pred_label': preds[test_ids]})

os.makedirs('submissions', exist_ok=True)
sub.to_csv('submissions/my_submission.csv', index=False)
print("✅ Saved 'submissions/my_submission.csv'. Upload this to GitHub!")