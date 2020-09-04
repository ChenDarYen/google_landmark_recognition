import torch
import torch.nn as nn
import time

import efficientnet
from utils.methods import *


def read_data(data_dir):
    pass


def train(epoch, data_loader, model, loss_func, optimizer, lr_scheduler):
    model.train()
    print('------ {} epoch ------'.format(epoch))

    batch_num = len(data_loader)
    start = time.time()
    losses = 0

    for i, data in enumerate(data_loader):
        batch_start = time.time()

        input, target = data
        input, target = input.cuda(), target.cuda()

        output = model(input.cuda())
        loss = loss_func(output, target)
        losses += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % LOG_STEPS == 0:
            print('{} / {}:'.format(i, batch_num),
                  'time {} ({})|'.format(batch_start - start, (time.time() - start) / (i + 1)),
                  'loss {} (P{])'.format(loss, losses / (i + 1)))


def inference(data_loader, model):
    model.eval()

    preds_list, confs_list = [], []

    with torch.no_grad():
        for i, input in enumerate(data_loader):
            input = input.cuda()
            output = model(input)

            preds, confs = torch.topk(output, NUM_PRED)
            preds_list.append(preds)
            confs_list.append(confs)

    return torch.cat(preds_list), torch.cat(confs_list)


def generate_submission(data_loader, model, label_encoder):
    preds, confs = inference(data_loader, model)
    preds, confs = preds.cpu(), confs.cpu()
    labels = [label_encoder.inverse_transform(pred) for pred in preds]

    sample_sub = pd.read_csv(TEST_CSV)
    sub = data_loader.dataset

    concat = lambda label, conf: ' '.join(['{} {}'.format(l, c) for l, c in zip(label, conf)])
    sub.landmarks = [concat(label, conf) for label, conf in zip(labels, confs)]

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv(TEST_CSV)


if __name__ == '__main__':
    classes_num, train_loader, test_loader, label_encoder = load_data(TRAIN_SCV, TEST_CSV, TRAIN_DIR, TEST_DIR)

    model = efficientnet.Model(COMPOUND_COEF, classes_num)

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

    print('inference')



