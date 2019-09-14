# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
from config import config
from torch import nn

succeed_flag = 1
failed_flag = -1
device = config.device
cuda_flag = config.cuda_flag
input_size = config.image_input_size
feature_extract = config.feature_extract


class ModelInference(object):
    def __init__(self, model):
        self.model = model

    '''
    batch data 预测
    '''
    def predict_image(self, image_batch_tensor):
        try:
            self.model.eval()
            if cuda_flag:
                image_batch_tensor = image_batch_tensor.cuda()
            outputs = self.model(image_batch_tensor)
            outputs_tensor = outputs.data
            if cuda_flag:
                outputs_tensor = outputs_tensor.cpu()
            m_softmax = nn.Softmax(dim=1)
            outputs_tensor = m_softmax(outputs_tensor)
            return succeed_flag, np.asarray(outputs_tensor)

        except Exception as e:
            print("predict error!", e)
            return failed_flag, []

    def predict_image_no_softmax(self, image_batch_tensor):
        try:
            self.model.eval()
            if cuda_flag:
                image_batch_tensor = image_batch_tensor.cuda()
            outputs = self.model(image_batch_tensor)
            outputs_tensor = outputs.data
            if cuda_flag:
                outputs_tensor = outputs_tensor.cpu()
            return succeed_flag, np.asarray(outputs_tensor)

        except Exception as e:
            print("predict error!", e)
            return failed_flag, []