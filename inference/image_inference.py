# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import cv2
import time
from PIL import Image
from config import config
from model import sport_model
from inference.model_inference import ModelInference
import torch
import torchvision.transforms as transforms
import numpy as np
from dataset.data_preprocess import ImageDataProcess
from torch.autograd import Variable
from torchvision.datasets.folder import default_loader

device = config.device
feature_extract = config.feature_extract


class SportModelInference(object):
    def __init__(self, pre_train_name, pre_train_path, model_path, class_nums=2):
        self.silent_model = sport_model.SportModel(class_nums, pre_train_name, pre_train_path, feature_extract).model.to(device)
        self.silent_model.load_state_dict(torch.load(model_path, map_location=device))
        self.silent_model_inference = ModelInference(self.silent_model)
        self.image_normalize = ImageDataProcess.image_normalize

    def test_image(self, img):
        img_tensor_list = []
        try:
            img_tensor = self.image_normalize(img)
            img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
        except:
            pass

        if len(img_tensor_list) > 0:
            try:
                input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
                flag, res_array = self.silent_model_inference.predict_image(input_tensor)
                return flag, res_array
            except:
                return -1, -1
        else:
            return -1, -1

