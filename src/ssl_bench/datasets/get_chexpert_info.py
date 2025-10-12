import pandas as pd
import numpy as np

EFFUSION_TRAIN_PATH = "/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/splits_effusion/train_labeled.csv"

def get_class_balance(file_path, label_col):
    df = pd.read_csv(file_path)  # keep header
    labels = df[label_col].astype(float)

    num_positive = np.sum(labels == 1.0)
    num_negative = np.sum(labels == 0.0)

    return num_positive, num_negative

pos, neg = get_class_balance(EFFUSION_TRAIN_PATH, label_col="Pleural Effusion")
print(f"Train Set EFFUSION - Positive: {pos}, Negative: {neg}")
