import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import multiprocessing
import time

import os

# dir
TRAIN_SCV = '../input/landmark-recognition-2020/train.csv'
TEST_CSV = '../input/landmark-recognition-2020/sample_submission.csv'
TRAIN_DIR = '../input/landmark-recognition-2020/train/'
TEST_DIR = '../input/landmark-recognition-2020/test/'

# general
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
CPU_NUM = multiprocessing.cpu_count()
LOG_STEPS = 100
MIN_SAMPLES_PER_CLASS = 143

# training
EPOCHS = 6
BATCH_SIZE = 64
INPUT_SIZE = 288

# model
COMPOUND_COEF = 1
DROPOUT_RATE = 0.25

# optimizer
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
LR = 0.001
STEP_SIZE = 1
GAMMA = 0.90

# inference
NUM_PRED = 20


class Dataset(Data.Dataset):
    def __init__(self, is_train, dataframe, data_dir):
        self.is_train = is_train
        self.dataframe = dataframe
        self.data_dir = data_dir

        if self.is_train:
            transforms_list = [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomApply([transforms.RandomResizedCrop(size=INPUT_SIZE)], p=0.33),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(
                        degrees=10, translate=(0.2, 0.2),
                        scale=(0.8, 1.2),
                        resample=Image.BILINEAR)
                ]),
                transforms.ToTensor(),
                transforms.RandomApply([transforms.RandomErasing(p=1, scale=(0.2, 0.33), ratio=(0.5, 2))], p=0.8),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
            ]
        else:
            transforms_list = [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
            ]
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        image_id = self.dataframe.iloc[index].id
        image_path = os.path.join(self.data_dir, image_id[0], image_id[1], image_id[2], '{}.jpg'.format(image_id))
        image = Image.open(image_path)
        image = self.transforms(image)

        if self.is_train:
            return [image, self.dataframe.iloc[index].landmark_id]
        else:
            return image

    def __len__(self):
        return self.dataframe.shape[0]


class Model(nn.Module):
    def __init__(self, compound_coef, classes_num):
        super(Model, self).__init__()
        self.compound_coef = compound_coef
        self.classes_num = classes_num

        self.base = EfficientNet.from_pretrained('efficientnet-b{}'.format(self.compound_coef))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        features_num = self.base._fc.in_features
        self.fc = nn.Linear(features_num, self.classes_num)

    def forward(self, inputs):
        bs = inputs.size(0)

        x = self.base.extract_features(inputs)
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


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
        num_workers=4,
    )
    test_loader = Data.DataLoader(
        dataset=Dataset(False, test, test_dir),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=CPU_NUM,
    )

    return classes_num, train_loader, test_loader, label_encoder


def train(epoch, data_loader, model, loss_func, optimizer, lr_scheduler):
    model.train()
    print('------ {} epoch ------'.format(epoch))

    batch_num = len(data_loader)
    start = time.time()
    losses = 0

    for i, data in enumerate(data_loader):
        batch_start = time.time()

        input_, target = data
        input_, target = input_.cuda(), target.cuda()

        output = model(input_)
        loss = loss_func(output, target)
        losses += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % LOG_STEPS == 0:
            print('{} / {}:'.format(i, batch_num),
                  'time {} ({}) | '.format(time.time() - batch_start, (time.time() - start) / (i + 1)),
                  'loss {} ({}) | '.format(loss, losses / (i + 1)),
                  'lr {}'.format(optimizer.param_groups[0]['lr']))

    lr_scheduler.step()


def inference(data_loader, model):
    model.eval()

    preds_list, confs_list = [], []

    with torch.no_grad():
        for i, input_ in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            input_ = input_.cuda()
            output = model(input_)

            confs, preds = torch.topk(output, NUM_PRED)
            preds_list.append(preds)
            confs_list.append(confs)

    return torch.cat(preds_list), torch.cat(confs_list)


def generate_submission(data_loader, model, label_encoder):
    preds, confs = inference(data_loader, model)
    preds, confs = preds.cpu().numpy(), confs.cpu().numpy()
    labels = [label_encoder.inverse_transform(pred) for pred in preds]

    sample_sub = pd.read_csv(TEST_CSV)
    sub = data_loader.dataset.dataframe

    concat = lambda label, conf: ' '.join(['{} {}'.format(l, c) for l, c in zip(label, conf)])
    sub.landmarks = [concat(label, conf) for label, conf in zip(labels, confs)]

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv('submission.csv')


if __name__ == '__main__':
    classes_num, train_loader, test_loader, label_encoder = load_data(TRAIN_SCV, TEST_CSV, TRAIN_DIR, TEST_DIR)

    model = Model(COMPOUND_COEF, classes_num)
    model.cuda()

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(
        params=model.parameters(),
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        lr=LR
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    for e in range(EPOCHS):
        train(e, train_loader, model, loss_func, optimizer, lr_scheduler)

    generate_submission(test_loader, model, label_encoder)
