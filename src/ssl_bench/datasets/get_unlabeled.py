import pandas as pd
from sklearn.model_selection import train_test_split
import os

csv_file = "/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/train_data_effusion.csv"
df = pd.read_csv(csv_file)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 12,000 labeled
labeled_df = df.iloc[:12000].copy()

# train/val/test (80/10/10)
train_df, temp_df = train_test_split(labeled_df, test_size=0.20, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

# 120,000 unlabeled
unlabeled_df = df.iloc[12000:132000].copy()
unlabeled_df = unlabeled_df[["Path"]]  # drop labels

out_dir = "/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/splits_effusion"
os.makedirs(out_dir, exist_ok=True)

train_df.to_csv(f"{out_dir}/train_labeled.csv", index=False)
val_df.to_csv(f"{out_dir}/val_labeled.csv", index=False)
test_df.to_csv(f"{out_dir}/test_labeled.csv", index=False)
unlabeled_df.to_csv(f"{out_dir}/unlabeled.csv", index=False)

print("Saved splits:")
print(f"  Train labeled: {len(train_df)}")
print(f"  Val labeled:   {len(val_df)}")
print(f"  Test labeled:  {len(test_df)}")
print(f"  Unlabeled:     {len(unlabeled_df)}")
