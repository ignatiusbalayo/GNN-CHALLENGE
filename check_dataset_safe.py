"""
Sanitized Dataset Diagnostic
Checks integrity WITHOUT revealing sensitive information
"""

import pandas as pd

print("=== SANITIZED DATASET CHECK ===")

train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test_no_label.csv')
private = pd.read_csv('data/private_labels.csv')

# 1. Check sizes match expected proportions
total = len(train) + len(val) + len(test)
train_pct = len(train) / total * 100
val_pct = len(val) / total * 100
test_pct = len(test) / total * 100

print(f"\n1. Split proportions:")
print(f"   Train: {train_pct:.0f}%")
print(f"   Val:   {val_pct:.0f}%")
print(f"   Test:  {test_pct:.0f}%")
print(f"   ✓ Looks reasonable" if 55 <= train_pct <= 65 else "   ⚠ Unusual split")

# 2. Feature dimensions (safe to show)
train_feats = len(train.columns) - 2
val_feats = len(val.columns) - 2
test_feats = len(test.columns) - 1

print(f"\n2. Feature dimensions:")
print(f"   All files have same features: {train_feats == val_feats == test_feats}")

# 3. Critical issue: ID mismatch
print(f"\n3. ID alignment:")
print(f"   test_no_label.csv has:  {len(test)} rows")
print(f"   private_labels.csv has: {len(private)} rows")
if len(test) == len(private):
    print(f"   ✅ MATCH - Good!")
else:
    print(f"   ❌ MISMATCH - This will cause grader to fail!")

# 4. Check ID overlaps (critical issue)
train_ids = set(train['id'])
val_ids = set(val['id'])
test_ids = set(test['id'])
private_ids = set(private['id'])

overlaps_exist = (
    len(train_ids & val_ids) > 0 or
    len(train_ids & test_ids) > 0 or
    len(val_ids & test_ids) > 0
)

print(f"\n4. ID overlaps between splits:")
if overlaps_exist:
    print(f"   ❌ OVERLAP DETECTED - Same IDs in multiple splits!")
else:
    print(f"   ✅ No overlaps - Good!")

# 5. Test vs Private alignment
print(f"\n5. Test-Private ID alignment:")
if test_ids == private_ids:
    print(f"   ✅ Perfect match - All test IDs have private labels!")
else:
    only_test = len(test_ids - private_ids)
    only_private = len(private_ids - test_ids)
    print(f"   ❌ Mismatch:")
    print(f"      {only_test} IDs in test but not in private_labels")
    print(f"      {only_private} IDs in private_labels but not in test")

# 6. Label format check (safe to verify format)
print(f"\n6. Label format:")
sample_labels = train['label'].unique()[:3]
all_start_with_class = all(str(label).startswith('Class_') for label in sample_labels)
if all_start_with_class:
    print(f"   ✅ Labels are anonymized (Class_X format)")
else:
    print(f"   ⚠ Labels might not be anonymized")

print(f"\n=== DIAGNOSIS ===")
if len(test) != len(private):
    print("❌ MAIN ISSUE: test_no_label.csv and private_labels.csv don't match")
    print("   FIX: Regenerate dataset OR manually align the files")
elif test_ids != private_ids:
    print("❌ MAIN ISSUE: test_no_label.csv has wrong IDs")
    print("   FIX: Filter test_no_label.csv to only include IDs in private_labels.csv")
elif overlaps_exist:
    print("❌ ISSUE: Some IDs appear in multiple splits")
    print("   FIX: Regenerate dataset with proper split separation")
else:
    print("✅ Dataset looks good! Ready to test baseline.")

print("\n" + "="*50)