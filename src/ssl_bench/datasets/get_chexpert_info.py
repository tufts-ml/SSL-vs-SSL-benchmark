import pandas as pd
import numpy as np

EFFUSION_TEST_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/test_data_effusion.csv"
EFFUSION_VAL_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/val_data_effusion.csv"
EFFUSION_TRAIN_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/train_data_effusion.csv"


ATELECTASIS_TEST_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/test_data_atelectasis.csv"
ATELECTASIS_VAL_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/val_data_atelectasis.csv"
ATELECTASIS_TRAIN_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/train_data_atelectasis.csv"

def get_class_balance(file_path, name):
    df = pd.read_csv(file_path)
    labels = df[name].astype(float)

    num_positive = np.sum(labels == 1.0)
    num_negative = np.sum(labels == 0.0)

    return num_positive, num_negative


for name, path in [("Train", EFFUSION_TRAIN_PATH),
                   ("Validation", EFFUSION_VAL_PATH),
                   ("Test", EFFUSION_TEST_PATH)]:
    pos, neg = get_class_balance(path, "Pleural Effusion")
    print(f"{name} Set EFFUSION - Positive: {pos}, Negative: {neg}")


for name, path in [("Train", ATELECTASIS_TRAIN_PATH),
                   ("Validation", ATELECTASIS_VAL_PATH),
                   ("Test", ATELECTASIS_TEST_PATH)]:
    pos, neg = get_class_balance(path, "Atelectasis")
    print(f"{name} Set ATELECTASIS - Positive: {pos}, Negative: {neg}")
