# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from config import config
from torch.nn import functional as F
device = config.device


class LeNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # 为什么要做reshape求说明
        out = self.fc(out)
        return out


class AutoSportModel(object):
    def __init__(self, layer_num, num_classes=5, model_name="inception", model_path="", feature_extract=True,):
        if model_name == "inception":
            self.model = models.inception_v3()
            self.model.load_state_dict(torch.load(model_path))
            self.set_parameter_requires_grad(self.model, feature_extract)
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 299
        elif model_name == "resnet18":
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224
        elif model_name == "resnet34":
            self.model = models.resnet34()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet50":
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet101":
            self.model = models.resnet101()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet152":
            self.model = models.resnet152()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet":
            self.model = models.mobilenet_v2()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, num_classes),
            )
        elif model_name == "shufflenet":
            self.model = models.shufflenet_v2_x1_0()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("Your pretrain model name is wrong!")

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


class SportModel(object):
    def __init__(self, num_classes=2, model_name="inception", model_path="", feature_extract=True):
        if model_name == "inception":
            self.model = models.inception_v3()
            self.model.load_state_dict(torch.load(model_path))
            self.set_parameter_requires_grad(self.model, feature_extract)
            # Handle the auxilary net
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 299
        elif model_name == "resnet18":
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 224
        elif model_name == "resnet34":
            self.model = models.resnet34()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet50":
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet101":
            self.model = models.resnet101()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet152":
            self.model = models.resnet152()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet":
            self.model = models.mobilenet_v2()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, num_classes),
            )
        elif model_name == "shufflenet":
            self.model = models.shufflenet_v2_x1_0()
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("Your pretrain model name is wrong!")

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    # model = LstmModel()
    # test_tensor = torch.randn((32, 30, 512))
    # print(test_tensor.size())
    # tt = model(test_tensor)
    # print(tt)
    a = 1