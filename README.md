# 🎓 GNN-Challenge: Neural Citation Network

Predict the research topic of a scientific paper using Graph Neural Networks(GNNs).

# 📌 Task Description

Given a citation graph where nodes represent scientific papers and edges represent citation relationships, the task is to predict the research topic of each paper.

Graph: Directed citation network

Nodes: Papers

Edges: Citations

Classes: 6 research topics

Some nodes are unlabeled and must be predicted during evaluation.

- **Nodes:** Papers  
- **Edges:** Citations  
- **Classes:** 6 research topics

## 📊 Data

The dataset is built from CiteSeer:
- `data/train.csv`: training nodes + labels
- `data/val.csv`: validation nodes + labels
- `data/test_no_label.csv`: test nodes (hidden labels)
- `data/edges.csv`: graph edges (citations)

## Dataset Column Descriptions
| **ID** | **feat₀** | **feat₁** | **feat₂** | … | **featₙ** | **Label** |
|:-----:|:---------:|:---------:|:---------:|:-:|:---------:|:--------:|
| 10    | 0.23      | 1.05      | 0.78      | … | 0.42      | **4**    |
| 25    | 0.62      | 0.18      | 0.91      | … | 0.37      | **1**    |
| 98    | 0.31      | 0.55      | 0.13      | … | 0.29      | **3**    |
| 102   | 0.35      | 0.69      | 0.47      | … | 0.12      | —        |
| 150   | 0.09      | 1.22      | 0.88      | … | 0.55      | —        |

**Notes:**
- Each row represents a paper node.
- `feat₀` … `featₙ` are continuous node features.
- `Label` denotes the research topic class.
- `—` indicates **unlabeled nodes** used for inference or evaluation.

Evaluation uses:
- Weighted F1 Score
- Per-class performance
- Overall accuracy (for quick feedback)
