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

train_type = 'yoga'


'''
模型训练参数
'''
#训练数据保存路径
config_data_path = os.path.join(project_path, "data", train_type)

#是否finetune（true：不finetune）
feature_extract = False

batch_size = 64
epoch_nums = 20

pre_train_model_name = "shufflenet"
# 模型的图片输入尺寸
input_size_dict = {"inception":299, "resnet18":224, "resnet34":224, "resnet50":224, "resnet101":224, "resnet152":224, "CNN":224, "LeNet": 28, "mobilenet":224, "shufflenet": 224}
image_input_size = input_size_dict[pre_train_model_name]
pre_train_model_path = os.path.join(project_path, "model/pre_train_model/%s.pth" % pre_train_model_name)

# 训练模型保存路径
config_model_save_path = os.path.join(project_path, "saved/model/individual/", train_type)
golf_label_dict = {"backswing": 0, "final": 1}
golf_ske_label_dict = {"backswing": 0, "standing": 1, "finish": 2}
# yoga_label_dict = {"boat-left": 0, "boat-right": 1, "camel-left": 2, "camel-right": 3, "chair-left": 4, "chair-right": 5,
#                    "cow-left": 6, "cow-right": 7, "tree_left": 8, "tree_right": 9, "triangle_left": 10,
#                    "triangle_right": 11}
yoga_label_dict = {"boat-left": 0, "boat-right": 1, "chair-left": 2, "chair-right": 3,
                   "tree_left": 4, "tree_right": 5, "triangle_left": 6,
                   "triangle_right": 7}
test_label_dict = {"class1": 0, "class2": 1, "class3": 2}
yoga_separate_dict = {"boat-left": 0, "boat-right": 1, "bow-left": 2, "bow-right": 3, "bridge-left": 4, "bridge-right": 5,
                      "camel-left": 6, "camel-right": 7, "chair-left": 8, "chair-right": 9, "cobra-left": 10,
                      "cobra-right": 11, "cow-left": 12, "cow-right": 13}
yoga_bothlr_dict = {"boat": 0, "bow": 1, "bridge": 2, "camel": 3, "chair": 4, "cobra": 5, "cow": 6}

if train_type == "golf":
    img_label_dict = golf_label_dict
elif train_type == "yoga":
    img_label_dict = yoga_label_dict
elif train_type == "golf_ske":
    img_label_dict = golf_ske_label_dict
elif train_type == "yoga_separate":
    img_label_dict = yoga_separate_dict
elif train_type == "yoga_bothlr":
    img_label_dict = yoga_bothlr_dict
else:
    raise ValueError("Your type is wrong. Please check again")

train_class_nums = len(img_label_dict)


# 自动训练参数
auto_train_type = 'yoga'
autotrain_folder = "1030"
val_ratio_ls = [0.1, 0.15, 0.2, 0.25, 0.3]
epoch_ls = [20]
pre_train_ls = ["shufflenet"]
learning_rate_ls = [0.001, 0.0005, 0.00025]

auto_golf_label_dict = {"backswing": 0, "final": 1}
# auto_yoga_label_dict = {"boat-left": 0, "boat-right": 1, "camel-left": 2, "camel-right": 3, "chair-left": 4,
#                         "chair-right": 5, "cow-left": 6, "cow-right": 7, "tree_left": 8, "tree_right": 9,
#                         "triangle_left": 10, "triangle_right": 11}
auto_yoga_label_dict = {"boat-left": 0, "boat-right": 1, "chair-left": 2, "chair-right": 3, "tree_left": 4,
                        "tree_right": 5, "triangle_left": 6, "triangle_right": 7}
auto_golf_ske_label_dict = {"backswing": 0, "finish": 1, "standing": 2}
auto_yoga_separate_dict = {"boat-left": 0, "boat-right": 1, "bow-left": 2, "bow-right": 3, "bridge-left": 4,
                           "bridge-right": 5, "camel-left": 6, "camel-right": 7, "chair-left": 8, "chair-right": 9,
                           "cobra-left": 10, "cobra-right": 11, "cow-left": 12, "cow-right": 13}
auto_yoga_bothlr_dict = {"boat": 0, "bow": 1, "bridge": 2, "camel": 3, "chair": 4, "cobra": 5, "cow": 6}

if auto_train_type == "golf":
    auto_train_label_dict = auto_golf_label_dict
elif auto_train_type == "yoga":
    auto_train_label_dict = auto_yoga_label_dict
elif auto_train_type == "golf_ske":
    auto_train_label_dict =  auto_golf_ske_label_dict
elif auto_train_type == "yoga_separate":
    auto_train_label_dict = auto_yoga_separate_dict
elif auto_train_type == "yoga_bothlr":
    auto_train_label_dict = auto_yoga_bothlr_dict
else:
    raise ValueError("Wrong train type!")

auto_class_num = len(auto_train_label_dict)


# 自动测试参数
model_folder = 'test/model/1014'
log = os.path.join(model_folder, "log.txt")
autotest_threshold = 0.4
positive_sample = r''
negative_sample = r''


if __name__ == "__main__":
    print(project_path)
    print(config_data_path)
    print(cuda_flag)
    print(config_data_path)
