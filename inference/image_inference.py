# -*- coding:utf-8 -*-
from __future__ import print_function
from config import config
from model import sport_model
import torch
import numpy as np
from torch import nn
from dataset.data_preprocess import ImageDataProcess

device = config.device
feature_extract = config.feature_extract
input_size = config.image_input_size


class SportModelInference(object):
    def __init__(self, class_nums, pre_train_name, pre_train_path, model_path):
        self.sport_model = sport_model.SportModel(class_nums, pre_train_name, pre_train_path, feature_extract).model.to(device)
        self.sport_model.load_state_dict(torch.load(model_path, map_location=device))
        self.image_normalize = ImageDataProcess.image_array_normalize

    def predict(self, img):
        img_tensor_list = []
        img_tensor = self.image_normalize(img, size=input_size)
        img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
        if len(img_tensor_list) > 0:
            input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
            res_array = self.predict_image(input_tensor)
            return res_array

    def predict_image(self, image_batch_tensor):
        self.sport_model.eval()
        image_batch_tensor = image_batch_tensor.cuda()
        outputs = self.sport_model(image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return np.asarray(outputs_tensor)
