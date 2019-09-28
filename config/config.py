# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch

'''
基本参数
'''
project_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_flag = torch.cuda.is_available()

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]

train_type = 'golf_ske'


'''
模型训练参数
'''
#训练数据保存路径
config_data_path = os.path.join(project_path, "data", train_type)

#是否finetune（true：不finetune）
feature_extract = False

batch_size = 64
epoch_nums = 20

pre_train_model_name = "mobilenet"
# 模型的图片输入尺寸
input_size_dict = {"inception":299, "resnet18":224, "resnet34":224, "resnet50":224, "resnet101":224, "resnet152":224, "CNN":224, "LeNet": 224, "mobilenet":224, "shufflenet": 224}
image_input_size = input_size_dict[pre_train_model_name]
pre_train_model_path = os.path.join(project_path, "model/pre_train_model/%s.pth" % pre_train_model_name)

# 训练模型保存路径
config_model_save_path = os.path.join(project_path, "saved/model/individual/", train_type)
golf_label_dict = {"backswing": 0, "final": 1}
golf_ske_label_dict = {"backswing": 0, "standing": 1, "finish": 2}
yoga_label_dict = {"tree":0, "crescent": 1}
test_label_dict = {"class1": 0, "class2":1, "class3":2}

if train_type == "golf":
    img_label_dict = golf_label_dict
elif train_type == "yoga":
    img_label_dict = yoga_label_dict
elif train_type == "golf_ske":
    img_label_dict = golf_ske_label_dict
else:
    raise ValueError("Your type is wrong. Please check again")

train_class_nums = len(img_label_dict)


# 自动训练参数
auto_train_type = 'golf_ske'
autotrain_folder = "0928"
val_ratio_ls = [0.2]
epoch_ls = [25]
pre_train_ls = ["mobilenet"]
learning_rate_ls = [0.001]

auto_golf_label_dict = {"backswing": 0, "final": 1}
auto_yoga_label_dict = {"tree": 0, "crescent": 1}
auto_golf_ske_label_dict = {"backswing": 0, "standing": 1, "finish":2}

if auto_train_type == "golf":
    auto_train_label_dict = auto_golf_label_dict
elif auto_train_type == "yoga":
    auto_train_label_dict = auto_yoga_label_dict
elif auto_train_type == "golf_ske":
    auto_train_label_dict =  auto_golf_ske_label_dict
else:
    raise ValueError("Wrong train type!")

auto_class_num = len(auto_train_label_dict)

# 自动测试参数
model_folder = 'test/model/0826'
log = os.path.join(model_folder, "log.txt")
autotest_threshold = 0.4
positive_sample = r''
negative_sample = r''


if __name__ == "__main__":
    print(project_path)
    print(config_data_path)
    print(cuda_flag)
    print(config_data_path)
