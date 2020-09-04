import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from consts import *


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

        x = self.base(inputs)
        x = self.avg_pool(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
