from inference.image_inference import SportModelInference
import os
import cv2
from openpyxl import Workbook
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_dict = ["backswing", "standing", "followthrough"]
num_class = len(class_dict)


class ModelTester(object):
    def __init__(self, m_path, s_path, r_path):
        self.path = m_path
        self.pre_model_name = ''
        self.model_type = ''
        self.pre_model_path = ''
        self.model = ''
        self.sample_name_list = []
        self.sample_path = s_path
        self.test_image_list = []
        self.flag = 0
        self.score = 0
        self.video_name = ''
        self.positive_record = []
        self.negative_record = []
        self.record = []
        self.record_path = r_path
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
        self.model = SportModelInference(self.num_class, self.pre_model_name, self.pre_model_path, self.path)

    def test_img(self, sample):
        for video_name in os.listdir(sample):
            self.video_name = video_name
            image = cv2.imread(os.path.join(sample, video_name))
            print(self.video_name)
            self.flag, self.score = self.model.test_image(image)
            idx = self.score[0].tolist().index(max(self.score[0].tolist()))
            print("Predicted action is: {}".format(class_dict[idx]))
            self.record_result()

    def record_result(self):
        record = []
        record.append(self.video_name)
        if type(self.score) == int:
            for num in range(self.num_class):
                record.append(-1)
        else:
            for item in self.score[0]:
                record.append(item)
        self.record.append(record)

    def write_result(self):
        wb = Workbook()
        ws = wb.create_sheet("result")
        ws.append(["video name", "abnormal", "normal"])
        for res in self.positive_record:
            try:
                ws.append([res[0], res[1], res[2]])
            except IndexError:
                print(res)
        wb.save(filename=self.record_path)

    def get_model_info(self):
        self.get_pretrain()
        self.load_model()

    def test_model(self):
        self.get_model_info()
        self.test_img(self.sample_path)
        # self.write_result()


if __name__ == "__main__":
    model_path = "test/model/golf_ske_shufflenet_2019-10-11-12-42-10.pth"
    sample_path = 'test/test1'
    record_path = model_path.replace(".pth", '_result.xlsx')
    Tester = ModelTester(model_path, sample_path, record_path)
    Tester.test_model()


