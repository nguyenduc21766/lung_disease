import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

# 1. Build metadata from folder structure
# ---------------------------------------------------------------------
# Root folder that contains NORMAL/ and PNEUMONIA/
MAIN_DIR = "data"

# Map folder names to numeric labels
CLASS_MAP = {
    "Normal": 0,
    "Pneumonia": 1,
}

records = []

for class_name, class_id in CLASS_MAP.items():
    class_dir = os.path.join(MAIN_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"Warning: directory not found: {class_dir}")
        continue

    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)

        # skip non-files just in case
        if not os.path.isfile(fpath):
            continue

        records.append({
            "Name": fname,
            "Path": fpath,
            "Label": class_id,   # 0 = Normal, 1 = Pneumonia
        })

metadata = pd.DataFrame(records)
print("Total images found:", len(metadata))
print(metadata.head())

# 2. Split: 80% train_val, 20% test (stratified)
# ---------------------------------------------------------------------
metadata["Label"] = metadata["Label"].astype(int)

train_val, test = train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata["Label"],
    random_state=42
)

print("Train+Val size:", len(train_val))
print("Test size:", len(test))

# 3. Stratified 5-fold on train_val
# ---------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

output_dir = os.path.join("data", "CSVs")
os.makedirs(output_dir, exist_ok=True)

# Save the full splits (optional but handy)
train_val.to_csv(os.path.join(output_dir, "train_val.csv"), index=False)
test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, train_val["Label"]), start=1):
    train_fold = train_val.iloc[train_idx]
    val_fold = train_val.iloc[val_idx]

    # Save folds
    train_fold.to_csv(os.path.join(output_dir, f"fold_{fold}_train.csv"), index=False)
    val_fold.to_csv(os.path.join(output_dir, f"fold_{fold}_val.csv"), index=False)

    # OPTIONAL: plot class distribution for sanity check
    """
    for df, name in [(train_fold, "train"), (val_fold, "val")]:
        df["Label"].value_counts().sort_index().plot(kind="bar")
        plt.title(f"Class distribution - {name} fold {fold}")
        plt.xlabel("Class (0=Normal, 1=Pneumonia)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"class_dist_{name}_fold{fold}.png")
        plt.close()
    """

print("CSV files written to:", output_dir)
