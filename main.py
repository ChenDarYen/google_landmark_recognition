import os

os.system('pip install efficientnet_pytorch')

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import multiprocessing
import time
import math

# dir
CHECKPOINT_DIR = '../input/landmark-recognition-2020-checkpoints/'
TRAIN_SCV = '../input/landmark-recognition-2020/train.csv'
TEST_CSV = '../input/landmark-recognition-2020/sample_submission.csv'
TRAIN_DIR = '../input/landmark-recognition-2020/train/'
TEST_DIR = '../input/landmark-recognition-2020/test/'

# general
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None
CPU_NUM = multiprocessing.cpu_count()
LOG_STEPS = 100
CLASSES_NUM = 1000

# training
EPOCHS = 5
BATCH_SIZE = 64
INPUT_SIZE = 288

# model
COMPOUND_COEF = 1
DROPOUT_RATE = 0.25
EMBEDDING_SIZE = 512

# optimizer
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
LR = 0.001
STEP_SIZE = 2
GAMMA = 0.91

# inference
MAX_NUM_PRED = 3


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


class SwishFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        y = x * torch.sigmoid(beta * x)
        ctx.save_for_backward(x, y, beta)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, beta = ctx.saved_tensors

        grad_x = grad_output * (beta * y + torch.sigmoid(beta * x) * (1 - beta * y))
        grad_beta = grad_output * (x * y - y ** 2)

        return grad_x, grad_beta


class MishFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x * torch.tanh(F.softplus(x))

        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors

        grad_x = grad_output * (x * torch.sigmoid(x) / torch.pow(torch.cosh(F.softplus(x)), 2) + y / x)
        return grad_x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.FloatTensor([1]))  # beta is initialized to be 1


    def forward(self, x):
        return SwishFunc.apply(x, self.beta)


class Mish(nn.Module):
    def forward(self, x):
        return MishFunc.apply(x)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv, self).__init__()
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.depthwise = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99)
        self.swish = Swish()

    def forward(self, inputs):
        x = self.pointwise(inputs)
        x = self.depthwise(x)
        x = self.swish(self.bn(x))

        return x


class Model(nn.Module):
    def __init__(self, compound_coef):
        super(Model, self).__init__()
        self.compound_coef = compound_coef

        self.base = EfficientNet.from_name('efficientnet-b{}'.format(self.compound_coef))
        features_num = self.base._fc.in_features
        print('features: ', features_num)
        self.separable = SeparableConv(features_num, 1792)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.fc = nn.Linear(1792, CLASSES_NUM)

    def forward(self, input):
        bs = input.size(0)

        x = self.extract_features(input)
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def extract_features(self, input):
        x = self.base.extract_features(input)
        x = self.separable(x)

        return x


# focal loss with label smoothing
class FocalLoss(nn.Module):
    def __init__(self, gamma, eps=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, label):
        smoothed_label = torch.ones_like(output) * (self.eps / CLASSES_NUM)
        for b, l in enumerate(label):
            smoothed_label[b, l] = 1 - self.eps

        pred = self.softmax(output)
        cross = -torch.pow(torch.abs(label - pred), self.gamma) * label * torch.log(pred)
        threshold = torch.ones_like(cross) * 100
        cross = torch.where(cross > 100, threshold, cross)
        loss = torch.mean(cross)

        return loss


def load_data(train_csv, test_csv, train_dir, test_dir):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    counts = train.landmark_id.value_counts()
    selected_classes = counts[:1000].index

    train = train.loc[train.landmark_id.isin(selected_classes)]

    label_encoder = LabelEncoder()
    label_encoder.fit(train.landmark_id.values)
    assert len(label_encoder.classes_) == CLASSES_NUM

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

    return train_loader, test_loader, label_encoder


def train(epoch, data_loader, model, criterion, optimizer, lr_scheduler):
    model.train()
    print('------ {} epoch ------'.format(epoch))

    batch_num = len(data_loader)
    start = time.time()
    losses = 0

    for i, data in enumerate(data_loader):
        batch_start = time.time()

        input, label = data
        input, label = input.cuda(), label.cuda()

        output = model(input)
        loss = criterion(output, label)

        losses += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % LOG_STEPS == 0:
            print('{} / {}: '.format(i, batch_num),
                  'time {} ({}) | '.format(time.time() - batch_start, (time.time() - start) / (i + 1)),
                  'loss {} ({}) | '.format(loss, losses / (i + 1)),
                  'lr {}'.format(optimizer.param_groups[0]['lr']))

    lr_scheduler.step()


def inference(data_loader, model):
    model.eval()

    preds_list, confs_list = [], []
    activation = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, input_ in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            if i > 6:
                break
            input_ = input_.cuda()
            output = model(input_)
            output = activation(output)

            confs, preds = torch.topk(output, MAX_NUM_PRED)
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
    train_loader, test_loader, label_encoder = load_data(TRAIN_SCV, TEST_CSV, TRAIN_DIR, TEST_DIR)

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'checkpoint_5.csv'))

    model = Model(COMPOUND_COEF)
    model.load_state_dict(checkpoint['net'])
    model.cuda()

    criterion = nn.CrossEntropyLoss()

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
    optimizer.load_state_dict(checkpoint['optim'])
    lr_scheduler.load_state_dict(checkpoint['lr_sch'])

    for e in range(EPOCHS):
        train(e, train_loader, model, criterion, optimizer, lr_scheduler)

    state = {
        'net': model.state_dict(),
        'optim': optimizer.state_dict(),
        'lr_sch': lr_scheduler.state_dict(),
        'epoch': checkpoint['epoch'] + EPOCHS,
    }
    torch.save(state, 'checkpoint_{}.csv'.format(state['epoch']))

    generate_submission(test_loader, model, label_encoder)
