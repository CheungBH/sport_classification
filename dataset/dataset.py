from __future__ import print_function
import torch
from config import config
from torch.utils.data import Dataset
from dataset.data_preprocess import ImageDataProcess
import os

input_size = config.image_input_size
data_dir = config.config_data_path
batch_size = config.batch_size
device = config.device


class DrownDataset(Dataset):
    def __init__(self, img_path, image_label_dict, image_processor=ImageDataProcess.image_normalize):
        self.img_dir_name = []
        self.img_dir_label = []
        for k, v in image_label_dict.items():
            self.img_dir_name.append(os.path.join(img_path, k))
            self.img_dir_label.append(v)

        self.img_name = []
        self.img_label = []

        for dir_name, dir_label in zip(self.img_dir_name, self.img_dir_label):
            img_file_names = os.listdir(os.path.join(dir_name))
            for img_name in img_file_names:
                self.img_name.append(os.path.join(dir_name, img_name))
                self.img_label.append(dir_label)

        self.image_processor = image_processor

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        image_name = self.img_name[item]
        label = self.img_label[item]
        try:
            image_object = self.image_processor(image_name)
            return image_object, label
        except Exception as e:
            print("error read ", image_name, e)
            # os.remove(image_name)
            image_name = self.img_name[0]
            _label = self.img_label[0]
            _image_object = self.image_processor(image_name)
            return _image_object, _label


class DataLoader(object):
    def __init__(self):
        self.image_datasets = {x: DrownDataset(os.path.join(data_dir, x), config.img_label_dict)
                               for x in ['train', 'val']}
        # Create training and validation dataloaders
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
                            for x in ['train', 'val',]}


class DataLoader_Auto(object):
    def __init__(self, data_src, label_dict, batch_size_auto):
        self.image_datasets = {x: DrownDataset(os.path.join(data_src, x), label_dict)
                               for x in ['train', 'val']}
        # Create training and validation dataloaders
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size_auto, shuffle=True, num_workers=1)
                            for x in ['train', 'val',]}