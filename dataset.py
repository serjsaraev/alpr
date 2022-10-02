import cv2
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A


class PlatesDataset(Dataset):

    def __init__(self, root_path: str, json_path: str, sample_type: str = 'train', val_size: float = 0.2,
                 random_state: int = 42, transform: A.Compose = None):

        self.root_path = root_path
        self.transfrom = transform

        with open(json_path, 'r') as f:
            img_list = json.load(f)

        if sample_type == 'train':
            self.img_list, _ = train_test_split(img_list, test_size=val_size, random_state=random_state)
        elif sample_type == 'val':
            _, self.img_list = train_test_split(img_list, test_size=val_size, random_state=random_state)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        try:
            img = cv2.imread(os.path.join(self.root_path, self.img_list[idx]['file']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            idx = 0
            img = cv2.imread(os.path.join(self.root_path, self.img_list[idx]['file']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print('Cant open image')

        objects = self.img_list[idx]['nums']

        bboxes = np.array([[min([i[0] for i in b['box']]), min([i[1] for i in b['box']]),
                            max([i[0] for i in b['box']]), max([i[1] for i in b['box']])] for b in objects])

        for bbox in bboxes:
            if (bbox[2] - bbox[0] < 0) or (bbox[3] - bbox[1] < 0):
                print(idx)

        labels = np.ones(shape=(bboxes.shape[0]), dtype=np.int64)

        try:
            if self.transfrom is not None:
                sample = self.transfrom(image=img, bboxes=bboxes, labels=labels)
                img, bboxes = sample['image'], sample['bboxes']
        except:
            print('Cant do augmention')

        img = transforms.ToTensor()(img)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        targets = {'boxes': bboxes, 'labels': labels, 'image_id': torch.as_tensor([idx])}

        return img, targets


def collate_fn(batch):
    return batch