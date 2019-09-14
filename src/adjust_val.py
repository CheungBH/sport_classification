import os
import random
import shutil


val_ratio = 0.3
class_mark = "screen_origin"


class ImgAdjuster(object):
    def __init__(self, val_r, src, mark):
        self.val_ratio = val_r
        self.data_src = src
        if os.listdir("../data"):
            self.train_src = os.path.join("../data", self.data_src, "train")
            self.val_src = os.path.join("../data", self.data_src, 'val')
        else:
            self.train_src = os.path.join("data", self.data_src, "train")
            self.val_src = os.path.join("data", self.data_src, 'val')
        self.type = os.listdir(self.train_src)
        self.train_path = ''
        self.val_path = ''
        self.train_ls = []
        self.val_ls = []
        self.class_mark = mark

    def adjust_img(self, class_type):
        self.train_path = os.path.join(self.train_src, class_type)
        self.val_path = os.path.join(self.val_src, class_type)
        self.makedir()
        self.train_ls = os.listdir(self.train_path)
        self.val_ls = os.listdir(self.val_path)
        total_num = len(self.train_ls) + len(self.val_ls)
        new_val_num = total_num * self.val_ratio
        dis = int(new_val_num - len(self.val_ls))
        if dis > 0:
            move_ls = random.sample(self.train_ls, abs(dis))
            for pic in move_ls:
                shutil.move(os.path.join(self.train_path, pic), self.val_path)
        else:
            move_ls = random.sample(self.val_ls, abs(dis))
            for pic in move_ls:
                try:
                    shutil.move(os.path.join(self.val_path, pic), self.train_path)
                except shutil.Error:
                    pass

    def makedir(self):
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.val_path, exist_ok=True)

    def run(self):
        print("Adjusting validation proportion now...")
        for c in self.type:
            if self.class_mark == "all":
                self.adjust_img(c)
                print("All the samples in {0} have been adjusted to {1} val successfully".format(self.data_src,
                                                                                                 self.val_ratio))
            else:
                if c == self.class_mark:
                    self.adjust_img(c)
                    print("Sample {0} in {1} have been adjusted to {2} val successfully".format(c, self.data_src,
                                                                                                self.val_ratio))
                else:
                    pass


if __name__ == '__main__':
    A = ImgAdjuster(val_ratio, "drown", class_mark)
    A.run()
