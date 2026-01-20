import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import pandas as pd
from dgl.nn import GraphConv

# 1. Define Model
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 2. Load Graph & Data
edges_df = pd.read_csv('data/edges.csv')
g = dgl.graph((edges_df['source'], edges_df['target']))
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test_no_label.csv')

# Prepping Features
# (Assuming features start after 'id' and 'label' columns)
feat_cols = [c for c in train_df.columns if c not in ['id', 'label']]
in_feats = len(feat_cols)
num_classes = train_df['label'].nunique()

# 3. Dummy Training Example
model = GCN(in_feats, 16, num_classes)
# ... (Student adds training loop here) ...

# 4. Generate Submission
model.eval()
with torch.no_grad():
    # Example: Simple inference on test nodes
    test_features = torch.FloatTensor(test_df[feat_cols].values)
    # Note: In a real GNN, you'd pass the full graph 'g'
    # logits = model(g, all_features)[test_df['id']]
    # For now, let's output a dummy CSV for testing the grader
    submission = pd.DataFrame({
        'id': test_df['id'],
        'label': 0 # Replace with: logits.argmax(1)
    })
    submission.to_csv('submissions/submission.csv', index=False)
    print("Done! Submission saved to submissions/submission.csv")