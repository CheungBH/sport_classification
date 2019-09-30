from config import config
from inference.image_inference import SportModelInference
import os
import cv2
import numpy as np
from openpyxl import Workbook
import time

device = config.device
model_folder = config.model_folder
log = os.path.join(model_folder, "log.txt")
threshold = config.autotest_threshold
positive_sample = config.positive_sample
negative_sample = config.negative_sample
num_class = config.auto_class_num


class AutoTester(object):
    def __init__(self, m_path, s_type, workbook):
        self.path = m_path
        self.model_name = self.path.split("/")[-1]
        self.folder = self.path.split("/")[-2]
        self.pre_model_name = ''
        self.model_type = ''
        self.pre_model_path = ''
        self.num_class = 0
        self.model = ''
        self.sample_name_list = []
        self.sample_type = s_type
        if self.sample_type == "positive":
            self.sample_path = positive_sample
        elif self.sample_type == "negative":
            self.sample_path = negative_sample
        else:
            raise ValueError("Wrong sample type!")
        self.test_image_list = []
        self.flag = 0
        self.score = []
        self.normal_index = 1
        self.video_name = ''
        self.record = []
        self.frame_len = 0
        self.result_path = ''
        self.wb = workbook
        self.detected_num = 0
        self.num_class = num_class

    def get_pretrain(self):
        if "_resnet18" in self.path:
            self.pre_model_name = "resnet18"
        elif "_resnet50" in self.path:
            self.pre_model_name = "resnet50"
        elif "_resnet34" in self.path:
            self.pre_model_name = "resnet34"
        elif "_resnet101" in self.path:
            self.pre_model_name = "resnet101"
        elif "_resnet152" in self.path:
            self.pre_model_name = "resnet152"
        elif "_inception" in self.path:
            self.pre_model_name = "inception"
        elif "_mobilenet" in self.path:
            self.pre_model_name = "mobilenet"
        elif "_shufflenet" in self.path:
            self.pre_model_name = "shufflenet"
        elif "_LeNet" in self.path:
            self.pre_model_name = "LeNet"
        else:
            raise ValueError("Wrong name of pre-train model")
        self.pre_model_path = os.path.join("model/pre_train_model/%s.pth" % self.pre_model_name)

    def load_model(self):
        self.model = SportModelInference(self.pre_model_name, self.pre_model_path, self.path, self.num_class)

    def test_img(self):
        for video_name in os.listdir(self.sample_path):
            try:
                self.test_image_list = []
                self.frame_len = min(config.max_frame_len, len(os.listdir(os.path.join(self.sample_path, video_name))))
                for i in range(self.frame_len):
                    test_frame_path = os.path.join(self.sample_path, "%s/%s_%d.jpg" % (video_name, video_name, i))
                    cv2_image_object = cv2.imread(test_frame_path)
                    if np.asarray(cv2_image_object).size == 0:
                        continue
                    self.test_image_list.append(cv2_image_object)
                if len(self.test_image_list) == 0:
                    continue
            except Exception as e:
                print(video_name, e)
                continue
            self.video_name = video_name
            self.model_pred()
            self.record_result()

    def model_pred(self):
        self.flag, self.score, self.detected_num = self.model.test_image(self.test_image_list)

    def record_result(self):
        record = []
        record.append(self.video_name)
        record.append(self.detected_num)
        if type(self.score) == int:
            for i in range(6):
                record.append(-1)
            print("{}  failed".format(self.video_name))
            record.insert(3, "failed")
        else:
            sum = 0
            for img_res in self.score:
                record.append(img_res[1])
                sum += img_res[1]
            average = sum/self.detected_num
            record.insert(2, average)
            for r in range(5 - self.detected_num):
                record.append('')
            # cnt = 0
            # result = "False"
            # for score in self.score:
            #     if score > threshold:
            #         cnt += 1
            #         result = "True"
            #         break
            #     else:
            #         continue
            # record.insert(1, cnt)
            # record.insert(4, cnt)
            if average > threshold:
                print("{}  True".format(self.video_name))
                record.insert(3, "True")
            else:
                print("{}  False".format(self.video_name))
                record.insert(3, "False")
        self.record.append(record)

    def write_result_to_excel(self):
        ws = self.wb.create_sheet(self.sample_type)
        ws.append(["video name", "total frame", "average score", "Pass condition", '1', '2'])
        for res in self.record:
            try:
                ws.append([res[0], res[1], res[2], res[3], res[4], res[5]])
            except IndexError:
                print(res)
        wb.save(filename=os.path.join(self.result_path, self.model_name[:-4] + '.xlsx'))
        print("Saving to {}".format(os.path.join(self.result_path, self.model_name[:-4] + '.xlsx')))
        self.wb.close()

    def write_result(self):
        self.result_path = os.path.join("test", "test_result", self.folder, self.model_type)
        # self.result_path = "test/test_sample"
        os.makedirs(self.result_path, exist_ok=True)
        self.write_result_to_excel()

    def get_model_info(self):
        self.get_pretrain()
        print("The pre_train model is {}".format(self.pre_model_name))
        self.load_model()

    def test_model(self):
        self.get_model_info()
        self.test_img()
        self.write_result()


if __name__ == "__main__":
    total_num = len(os.listdir(model_folder))
    cnt = 0
    for model_name in os.listdir(model_folder)[0:1]:
        cnt += 1
        model_path = model_folder + '/' + model_name
        file = open(log, "a")
        file.write("{0}/{1}\n".format(cnt, total_num))
        file.write("Testing model: {}\n\n".format(model_path))
        print("Testing model: {}".format(model_path))
        print("{0}/{1}".format(cnt, total_num))
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        wb = Workbook()
        for sample_type in ['negative', "positive"]:
            print("testing {} samples".format(sample_type))
            file.write("Starting time: {}\n".format(time_str))
            file.write("testing {} samples: ".format(sample_type))
            file.write(model_name + '\n')
            # try:
            Tester = AutoTester(model_path, sample_type, wb)
            Tester.test_model()
            file.write("Model have been tested successfully\n")
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            file.write("End time: {}\n\n".format(time_str))
        #     except:
        #         file.write("Errors occur when testing\n")
        #         time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        #         file.write("End time: {}\n\n".format(time_str))
        # file.close()
