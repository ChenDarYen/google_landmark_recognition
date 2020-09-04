import torch.utils.data as Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from dataset import Dataset
from consts import *


def load_data(train_csv, test_csv, train_dir, test_dir):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    counts = train.landmark_id.value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    classes_num = selected_classes.shape[0]
    print('{} classes with at least {} samples.'.format(classes_num, MIN_SAMPLES_PER_CLASS))

    train = train.loc[train.landmark_id.isin(selected_classes)]

    label_encoder = LabelEncoder()
    label_encoder.fit(train.landmark_id.values)
    assert len(label_encoder.classes_) == classes_num

    train.landmark_id = label_encoder.transform(train.landmark_id)

    train_loader = Data.DataLoader(
        dataset=Dataset(True, train, train_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=CPU_NUM,
    )
    test_loader = Data.DataLoader(
        dataset=Dataset(False, test, test_dir),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=CPU_NUM,
    )

    return classes_num, train_loader, test_loader, label_encoder
