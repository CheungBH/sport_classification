from model import sport_model
from trainer.trainer import ModelTrainer
from dataset import dataset
from config import config
import torch.nn as nn
import torch.optim as optim
import time
import os
from src import adjust_val

train_type = config.auto_train_type
mark = config.autotrain_folder

val_ratio_ls = config.val_ratio_ls
epoch_ls = config.epoch_ls
pre_train_ls = config.pre_train_ls
label_dict = config.auto_train_label_dict
class_num = len(label_dict)
feature_extract = config.feature_extract
device = config.device
learning_rate_ls = config.learning_rate_ls


class AutoTrainer(object):
    def __init__(self, pre_name, epo, lr):
        self.pre_train_model_name = pre_name
        self.criterion = nn.CrossEntropyLoss()
        self.data_src = os.path.join("data", train_type)
        self.data_loader = ''
        self.class_num = class_num
        self.pre_model_path = ''
        self.optimizer_ft = ''
        self.epoch = epo
        self.lr = lr
        self.time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.model_str = train_type + "_%s_%s.pth" % (self.pre_train_model_name, self.time_str)
        self.model_save_folder = os.path.join("saved/model/", mark, train_type, self.pre_train_model_name)
        self.log_save_path = os.path.join(self.model_save_folder, self.model_str.replace(".pth", "_log.txt"))
        self.model_save_path = os.path.join(self.model_save_folder, self.model_str)
        self.record_path = os.path.join("saved/model/", mark, train_type, "result.txt")
        self.is_inception = self.pre_train_model_name == "inception"
        self.size = 224
        self.batch_size = 64
        self.sport_model = ''

    def get_pretrain_info(self):
        self.pre_model_path = "model/pre_train_model/{}.pth".format(self.pre_train_model_name)
        if self.pre_train_model_name == "resnet50" or self.pre_train_model_name == "resnet101" or self.pre_train_model_name == "resnet152":
            self.batch_size = 32
        if self.pre_train_model_name == 'inception':
            self.size = 299
            self.batch_size = 32
        if self.pre_train_model_name == "LeNet":
            self.batch_size = 512
            self.size = 28

    def load_model(self):
        if self.pre_train_model_name == "LeNet":
            self.sport_model = sport_model.LeNet(self.class_num).to(device)
        else:
            self.sport_model = sport_model.SportModel(self.class_num, self.pre_train_model_name, self.pre_model_path, feature_extract).model.to(device)
        params_to_update = self.sport_model.parameters()
        self.optimizer_ft = optim.Adam(params_to_update, lr=self.lr)

    def record(self):
        file = open(self.record_path, 'a')
        file.write("Model name is: {}\n".format(self.model_str))
        file.write("Information\n")
        file.write("{} epochs\n".format(self.epoch))
        file.write("{} validation\n".format(val))
        file.write("The pre-train model is {}\n".format(self.pre_train_model_name))
        file.write("Trained successfully\n\n")
        file.close()

    def auto_train(self):
        self.get_pretrain_info()
        self.load_model()
        os.makedirs(self.model_save_folder, exist_ok=True)
        self.data_loader = dataset.DataLoader_Auto(self.data_src, label_dict, self.batch_size, self.size)
        # MT = trainer.ModelTrainer(train_type, self.silent_model, self.size)
        # MT.train_with_test(self.data_loader.dataloaders_dict, self.criterion, self.optimizer_ft, self.epoch, self.is_inception, self.model_save_path, self.log_save_path)
        ModelTrainer.train_sport_model(self.sport_model, self.data_loader.dataloaders_dict, self.criterion, self.optimizer_ft, self.epoch, self.is_inception, self.model_save_path, self.log_save_path)
        print("train model done, save model to %s" % os.path.join(self.model_save_path, self.model_str))
        self.record()

    def failed_record(self):
        file = open(self.record_path, 'a')
        file.write("Model name is: {}\n".format(self.model_str))
        file.write("Information\n")
        file.write("{} epochs\n".format(self.epoch))
        file.write("{} validation\n".format(val))
        file.write("The pre-train model is {}\n".format(self.pre_train_model_name))
        file.write("Something wrong happens when training\n")


if __name__ == "__main__":
    cnt = 0
    total_num = len(epoch_ls) * len(pre_train_ls) * len(val_ratio_ls) * len(learning_rate_ls)
    for val in val_ratio_ls:
        for cls in label_dict.keys():
            IA = adjust_val.ImgAdjuster(val, train_type, cls)
            IA.run()
        for pre_model in pre_train_ls:
            for epoch in epoch_ls:
                for learning_rate in learning_rate_ls:
                    cnt += 1
                    print("Beginning to train: {0}/{1}".format(cnt, total_num))
                    print("The validation proportion is {}".format(val))
                    print("It will train {} epochs".format(epoch))
                    print("The pre_train model is {}".format(pre_model))
                    AutoTrain = AutoTrainer(pre_model, epoch, learning_rate)
                    # try:
                    AutoTrain.auto_train()
                    # except:
                    #     AutoTrain.failed_record()
