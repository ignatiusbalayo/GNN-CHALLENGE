# 🎓 GNN-Challenge: Citation Network Analysis

## 🧠 The Objective
You are tasked with building a node classification model for the **CiteSeer** citation network. 
**The Catch:** You must compare a standard **GCN** against a **Graph Attention Network (GAT)**.

## 📚 The Data: CiteSeer
* **Nodes:** 3,327 Papers.
* **Edges:** 4,732 Citations.
* **Classes:** 6 Research Topics (AI, ML, DB, etc.).

## 🛠 Tasks & Lecture References

### Task 1: Establish a Baseline (GCN)
Run the provided `baseline.py`. It implements a 2-layer GCN.
* **Reference:** [Video 3.4: Graph Convolutional Networks (GCN)](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T&index=13)
* **Goal:** Record the Test Accuracy (Expect ~70%).

### Task 2: Implement Graph Attention (GAT)
Modify the code to use `GATConv` instead of `GraphConv`.
* **Reference:** [Video 3.5: Graph Attention Networks (GAT)](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T&index=14)
* **Theory:** How does the attention mechanism ($\alpha_{ij}$) change how neighbor information is aggregated?
* **Goal:** Beat the GCN baseline accuracy.

### Task 3: Hyperparameter Tuning
* **Reference:** [Video 4.2: Neighbor Sampling & Training](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T&index=18)
* Experiment with:
    * `num_heads` (for GAT).
    * `hidden_feats` (try 16, 32, 64).
    * `dropout` (try 0.5 to prevent overfitting).

## 🚀 How to Run
1. **Install:** `pip install -r requirements.txt`
2. **Run:** `python starter_code/baseline.py`

## 🚀 How to Complete the Challenge

1.  **Train your Model:**
    ```bash
    python starter_code/baseline.py
    ```
    *This will create a file called `submission.csv`.*

2.  **Get Your Grade:**
    ```bash
    python grade_me.py
    ```
    *This will verify your file against the true labels and tell you if you Passed or Failed.*

3.  **The Goal:**
    * **Level 1 (Pass):** > 65% Accuracy (GCN)
    * **Level 2 (Distinction):** > 75% Accuracy (GAT)

    Final instrcutions

    # 🧠 The Neural Citation Challenge

**Can you predict the topic of a scientific paper using Graph Neural Networks?**

## 🏁 How to Participate
1. **Fork** this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Train the baseline model: `python starter_code/baseline.py`.
   * This creates `submissions/my_submission.csv`.
4. **Submit:**
   * Commit your CSV file.
   * Open a **Pull Request** to the main repository.
   * 🤖 **Wait 30 seconds... The Auto-Grader will comment with your score!**

## 🎯 The Goal
* **Pass:** > 65% Accuracy.
* **Distinction:** > 75% Accuracy (Hint: Try implementing GAT).