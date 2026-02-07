import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Ensure output directory exists for GitHub Actions
os.makedirs('submissions', exist_ok=True)

print("=" * 50)
print("🚀 ALIGNED BASELINE - NetClass Arena")
print("=" * 50)

# 1. Load data
print("\n📥 Loading data...")
train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test_no_label.csv')

# 2. Prepare features and labels
# We use the column names we discovered: 'id' and 'label'
X_train = train.drop(['id', 'label'], axis=1)
y_train = train['label']

X_val = val.drop(['id', 'label'], axis=1)
y_val = val['label']

# 3. Align Features (The "Challenge" Step)
# Since test_no_label might have different feature column names or order,
# we ensure the test set matches the training column structure exactly.
feature_cols = X_train.columns
X_test = test[feature_cols] 
test_ids = test['id']

print(f"   Train: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"   Val:   {len(X_val)} samples")
print(f"   Test:  {len(X_test)} samples")

# 4. Handle Missing Values (Imputation)
# Even if Random Forest is robust, if the noise generation created NaNs, we fill them.
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# 5. Train baseline model
print("\n🏋️ Training Random Forest...")
# We use balanced class_weight to handle the imbalance you baked in
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate on validation
print("\n📊 Validation Results:")
y_pred_val = clf.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val) * 100
val_f1 = f1_score(y_val, y_pred_val, average='macro')

print(f"   Accuracy:  {val_acc:.2f}%")
print(f"   Macro-F1:  {val_f1:.4f}")

# 7. Make predictions on test set
print("\n🔮 Generating test predictions...")
test_preds = clf.predict(X_test)

# 8. Save submission
# IMPORTANT: We MUST keep the original 'id' from test_no_label.csv
submission = pd.DataFrame({
    'id': test_ids,
    'pred_label': test_preds # Aligning column name with grader expectations
})

submission.to_csv('submissions/baseline_submission.csv', index=False)

print(f"   ✅ Saved: submissions/baseline_submission.csv")
print("\n" + "=" * 50)
print("✅ COMPLETE!")