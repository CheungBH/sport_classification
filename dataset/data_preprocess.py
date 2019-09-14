# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import torch
from config import config

image_normalize_mean = config.image_normalize_mean
image_normalize_std = config.image_normalize_std


class ImageDataProcess(object):
    def __init__(self):
        pass

    @staticmethod
    def image_array_normalize(image_array, size=224):
        image_array = cv2.resize(image_array, (size, size))
        image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1))
        for channel, _ in enumerate(image_array):
            image_array[channel] /= 255.0
            image_array[channel] -= image_normalize_mean[channel]
            image_array[channel] /= image_normalize_std[channel]
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor

    @staticmethod
    def image_normalize(image, size=224):
        image_array = cv2.imread(image)
        image_array = cv2.resize(image_array, (size, size))
        image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
        image_array = image_array.transpose((2, 0, 1))
        for channel, _ in enumerate(image_array):
            image_array[channel] /= 255.0
            image_array[channel] -= image_normalize_mean[channel]
            image_array[channel] /= image_normalize_std[channel]
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor


if __name__ == "__main__":
    test_image_path = ""
    image_tensor = ImageDataProcess.image_normalize(test_image_path)
    print(type(image_tensor))
    print(image_tensor.shape)
