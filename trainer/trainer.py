# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy
from config import config
import codecs
from dataset.data_preprocess import ImageDataProcess
device = config.device


class ModelTrainer(object):
    def __init__(self, model_type, model, size):
        self.model_type = model_type
        self.model = model
        self.fake_sample = []
        if model_type == "attack":
            self.positive_folder = "data/test/face_whole/positive".format(self.model_type)
            self.attack_fake_folder = "data/test/attack/fake"
            for fake_name in os.listdir(self.attack_fake_folder):
                self.fake_sample.append(os.path.join(self.attack_fake_folder, fake_name))
        else:
            self.positive_folder = "data/test/{}/positive".format(self.model_type)
        self.negative_folder = "data/test/{}/negative".format(self.model_type)
        self.positive_sample = []
        self.negative_sample = []
        self.size = size
        self.image_normalize = ImageDataProcess.image_array_normalize
        self.positive_result = []
        self.negative_result = []
        self.fake_result = []
        self.pos_test_info = []
        self.neg_test_info = []
        self.fake_wrong = []
        self.attack_pos_threshold = 0.4
        self.result_writer = ''
        self.output_file = ''
        self.fake_threshold = 0.8
        for pos_name in os.listdir(self.positive_folder):
            self.positive_sample.append(os.path.join(self.positive_folder, pos_name))
        for neg_name in os.listdir(self.negative_folder):
            self.negative_sample.append(os.path.join(self.negative_folder, neg_name))
        self.img_name = ''
        self.model_save_path = ''

    def test_img(self, image):
        img_ls = []
        image_tensor = self.image_normalize(image, self.size)
        img_ls.append(torch.unsqueeze(image_tensor, 0))
        input_tensor = torch.cat(tuple(img_ls), dim=0)
        self.model.eval()
        input_tensor = input_tensor.cuda()
        outputs = self.model(input_tensor)
        outputs = outputs.cpu()
        # print(outputs)
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs)
        # print(outputs_tensor)
        return outputs_tensor

    def count_acc(self, ls):
        right_num = ls.count(1)
        wrong_num = ls.count(0)
        return right_num / len(ls), right_num, wrong_num

    def record_test_detail(self):
        self.result_writer.write("Negative\n")
        self.result_writer.write("Video_name\tpredicted_result\tpositive_score\n")
        for item in self.neg_test_info:
            self.result_writer.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        self.result_writer.write("\nPositive\n")
        self.result_writer.write("Video_name\tpredicted_result\tpositive_score\n")
        for item in self.pos_test_info:
            self.result_writer.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))

    def check_fake(self):
        fake_acc, a, b = self.count_acc(self.fake_result)
        if fake_acc < 0.8:
            return False
        else:
            return True

    def test_attack_positive(self):
        for pos_img in self.positive_sample:
            self.img_name = pos_img.split("\\")[-1]
            img = cv2.imread(pos_img)
            pos_output = self.test_img(img)
            pos_output = pos_output.cpu().data.numpy()
            self.positive_result.append(pos_output[0][1])
            self.pos_test_info.append([pos_img, np.argmax(pos_output, axis=1), pos_output[0][1]])

    def test_attack_negative(self):
        for neg_img in self.negative_sample:
            self.img_name = neg_img.split("\\")[-1]
            img = cv2.imread(neg_img)
            pos_output = self.test_img(img)
            pos_output = pos_output.cpu().data.numpy()
            self.negative_result.append(pos_output[0][1])
            self.neg_test_info.append([neg_img, np.argmax(pos_output, axis=1), pos_output[0][1]])

    def test_attack_fake(self):
        for fake_img in self.fake_sample:
            self.img_name = fake_img.split("\\")[-1]
            img = cv2.imread(fake_img)
            fake_output = self.test_img(img)
            fake_output = fake_output.cpu().data.numpy()
            index_f = np.argmax(fake_output, axis=1)
            if index_f == 1:
                self.fake_result.append(0)
                self.fake_wrong.append(self.img_name)
            else:
                self.fake_result.append(1)

    def attack_test(self):
        sorted_positive = sorted(self.positive_result)
        sorted_negative = sorted(self.negative_result)
        threshold = sorted_positive[
            len(self.positive_result) - int(len(self.positive_result) * self.attack_pos_threshold)]
        for score, cnt in zip(sorted_negative, range(len(sorted_negative))):
            if score > threshold:
                break
            else:
                cnt += 1
        acc = cnt / len(sorted_negative)
        if self.check_fake() and threshold > 0.1:
            return acc, threshold
        else:
            return 0, 0

    def test_positive(self):
        for pos_img in self.positive_sample:
            self.img_name = pos_img.split("\\")[-1]
            img = cv2.imread(pos_img)
            try:
                pos_output = self.test_img(img)
                pos_output = pos_output.cpu().data.numpy()
                index_p = np.argmax(pos_output, axis=1)
                self.pos_test_info.append([pos_img, index_p, pos_output[0][1]])
                if index_p == 1:
                    self.positive_result.append(1)
                else:
                    self.positive_result.append(0)
            except:
                self.positive_result.append(0)
                self.pos_test_info.append([pos_img, -1, "undetected"])

    def test_negative(self):
        for neg_img in self.negative_sample:
            img = cv2.imread(neg_img)
            try:
                neg_output = self.test_img(img)
                neg_output = neg_output.cpu().data.numpy()
                index_n = np.argmax(neg_output, axis=1)
                self.neg_test_info.append([neg_img, index_n, neg_output[0][1]])
                if index_n == 1:
                    self.negative_result.append(0)
                else:
                    self.negative_result.append(1)
            except:
                self.negative_result.append(1)
                self.neg_test_info.append([neg_img, -1, "undetected"])

    def train_with_test(self, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,
                        model_save_path="./", log_save_path=""):
        since = time.time()
        val_acc_history = []
        test_best_acc = 0
        result_path = log_save_path.replace(".txt", "_test_record.txt")
        img_folder = '/'.join(model_save_path.split('\\')[:-1]) + "/train_pic"
        self.model_save_path = model_save_path

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        log_file_writer = codecs.open(log_save_path, mode="w")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 20)
            log_file_writer.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
            log_file_writer.write('-' * 10 + "\n")

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print(phase)
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                batch_num = 0
                batch_start_time = time.time()
                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if is_inception and phase == 'train':
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if batch_num % 100 == 0:
                        print("batch num:", batch_num, "cost time:", time.time() - batch_start_time)
                        batch_start_time = time.time()
                    batch_num += 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                log_file_writer.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc

                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            if self.model_type != "attack":
                self.test_positive()
                self.test_negative()
                neg_test_acc, neg_right, neg_wrong = self.count_acc(self.negative_result)
                pos_test_acc, pos_right, pos_wrong = self.count_acc(self.positive_result)
                average_acc = (neg_test_acc + pos_test_acc) / 2
            else:
                self.test_attack_positive()
                self.test_attack_negative()
                self.test_attack_fake()
                average_acc, pos_threshold = self.attack_test()
                fake_test_acc, fake_right, fake_wrong = self.count_acc(self.fake_result)
                log_file_writer.write("The fake sample accuracy is {}\n".format(fake_test_acc))
                if self.fake_wrong == []:
                    log_file_writer.write("All the fake samples are right")
                else:
                    log_file_writer.write("Wrong samples are:\n")
                    for sample in self.fake_wrong:
                        log_file_writer.write(sample + '\t')
                log_file_writer.write('\n')
                log_file_writer.write("Negative accuracy is {}\n".format(average_acc))

            if average_acc > test_best_acc:
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, model_save_path)
                test_best_acc = average_acc
                log_file_writer.write("This is the latest version of model being saved\n")
                record_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                log_file_writer.write("saving time is {}\n".format(record_time))
                self.result_writer = open(result_path, "w")
                if self.model_type != "attack":
                    self.result_writer.write("The negative accuracy is {}\n".format(neg_test_acc))
                    self.result_writer.write("Wrong num:{0}\t\t""Right num:{1}\n".format(neg_wrong, neg_right))
                    self.result_writer.write("The positive accuracy is {}\n".format(pos_test_acc))
                    self.result_writer.write("Wrong num:{0}\t\t""Right num:{1}\n".format(pos_wrong, pos_right))
                    self.record_test_detail()
                    log_file_writer.write("The accuracy of negative test samples is: {}\n".format(neg_test_acc))
                    log_file_writer.write("The accuracy of positive test samples is: {}\n".format(pos_test_acc))
                else:
                    log_file_writer.write("The threshold is {}\n".format(pos_threshold))
                self.result_writer.write("The time saving the model is {}\n\n\n".format(record_time))

            self.negative_result = []
            self.positive_result = []
            self.pos_test_info = []
            self.neg_test_info = []
            self.fake_result = []
            self.fake_wrong = []

            epoch_time_cost = time.time() - epoch_start_time
            print("epoch complete in {:.0f}m {:.0f}s".format(epoch_time_cost // 60, epoch_time_cost % 60))
            log_file_writer.write(
                "epoch complete in {:.0f}m {:.0f}s\n\n".format(epoch_time_cost // 60, epoch_time_cost % 60))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))
        # print('Best test Acc: {:.4f}'.format(max(test_acc_history)))

        log_file_writer.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        log_file_writer.write('Best val Acc: {:.4f}\n'.format(best_acc))
        # log_file_writer.write('Best test Acc: {:.4f}\n'.format(max(test_acc_history)))

        log_file_writer.close()

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, val_acc_history

    @staticmethod
    def train_drown_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,
                           model_save_path="./", log_save_path=""):
        since = time.time()
        val_acc_history = []
        test_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        log_file_writer = codecs.open(log_save_path, mode="w")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 20)
            log_file_writer.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
            log_file_writer.write('-' * 10 + "\n")

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print(phase)
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                batch_num = 0
                batch_start_time = time.time()
                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if is_inception and phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if batch_num % 100 == 0:
                        print("batch num:", batch_num, "cost time:", time.time() - batch_start_time)
                        batch_start_time = time.time()
                    batch_num += 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                log_file_writer.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, model_save_path)

                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            epoch_time_cost = time.time() - epoch_start_time
            print("epoch complete in {:.0f}m {:.0f}s".format(epoch_time_cost // 60, epoch_time_cost % 60))
            log_file_writer.write(
                "epoch complete in {:.0f}m {:.0f}s\n".format(epoch_time_cost // 60, epoch_time_cost % 60))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))
        # print('Best test Acc: {:.4f}'.format(max(test_acc_history)))

        log_file_writer.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        log_file_writer.write('Best val Acc: {:.4f}\n'.format(best_acc))
        # log_file_writer.write('Best test Acc: {:.4f}\n'.format(max(test_acc_history)))

        log_file_writer.close()

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

