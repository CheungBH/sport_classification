# -*- coding:utf-8 -*-
from __future__ import print_function
from model import sport_model
from trainer import trainer
from dataset import dataset
from config import config
import torch.nn as nn
import torch.optim as optim
import time
import os

device = config.device
feature_extract = config.feature_extract
num_epochs = config.epoch_nums
model_save_path = config.config_model_save_path
class_nums = config.train_class_nums
pre_train_model_name = config.pre_train_model_name
pre_train_model_path = config.pre_train_model_path
cuda_flag = config.cuda_flag
model_type = config.train_type


if __name__ == "__main__":
    os.makedirs(model_save_path, exist_ok=True)
    if pre_train_model_name == "LeNet":
        model = sport_model.LeNet(class_nums).to(device)
    else:
        model = sport_model.SportModel(class_nums, pre_train_model_name, pre_train_model_path,
                                                       feature_extract).model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    data_loader = dataset.DataLoader()

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_str = model_type + "_%s_%s.pth" % (pre_train_model_name, time_str)
    log_save_path = os.path.join(model_save_path, model_str.replace(".pth", "_log.txt"))

    is_inception = pre_train_model_name == "inception"
    silent_detect_model, hist = trainer.ModelTrainer.train_sport_model(model, data_loader.dataloaders_dict, criterion,
                                                      optimizer_ft, num_epochs=num_epochs, is_inception=is_inception,
                                                                 model_save_path=os.path.join(model_save_path, model_str),
                                                                        log_save_path= log_save_path)

    # save model
    print("train model done, save model to %s" % os.path.join(model_save_path, model_str))
