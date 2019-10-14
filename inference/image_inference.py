# -*- coding:utf-8 -*-
from __future__ import print_function
from config import config
from model import sport_model
from inference.model_inference import ModelInference
import torch
from dataset.data_preprocess import ImageDataProcess

device = config.device
feature_extract = config.feature_extract


class SportModelInference(object):
    def __init__(self, class_nums, pre_train_name, pre_train_path, model_path):
        self.sport_model = sport_model.SportModel(class_nums, pre_train_name, pre_train_path, feature_extract).model.to(device)
        self.sport_model.load_state_dict(torch.load(model_path, map_location=device))
        self.sport_model_inference = ModelInference(self.sport_model)
        self.image_normalize = ImageDataProcess.image_array_normalize

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
                flag, res_array = self.sport_model_inference.predict_image(input_tensor)
                return flag, res_array
            except:
                return -1, -1
        else:
            return -1, -1

