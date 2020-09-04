import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

from consts import *

train = pd.read_csv('../input/landmark-recognition-2020/train.csv')
test = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
train_dir = '../input/landmark-recognition-2020/train/'
test_dir = '../input/landmark-recognition-2020/test/'


class Dataset(data.Dataset):
    def __init__(self, is_train, data_frame, data_dir):
        self.is_train = is_train
        self.data_frame = data_frame
        self.data_dir = data_dir

        if self.is_train:
            transforms_list = [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(size=INPUT_SIZE),
                    transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.5, 2))
                ]),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                                        transforms.RandomRotation(degrees=10)],
                                       p=0.5),
                transforms.ToTensor(),
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
        image_id = self.data_frame.iloc[index].id
        image_path = os.path.join(self.data_dir, image_id[0], image_id[1], image_id[2], '{}.jpg'.format(image_id))
        image = self.transforms(Image.open(image_path))

        if self.is_train:
            return [image, self.data_frame.iloc[index].landmark_id]
        else:
            return image

    def __len__(self):
        return self.data_frame.shape[0]
