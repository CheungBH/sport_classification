from utils.utils import Utils
import os

train_type = "golf_ske"
train_path = os.path.join("../data", train_type, "train")
val_path = os.path.join("../data", train_type, "val")


def delete(path):
    for cls in os.listdir(path):
        Utils.delete_wrong(os.path.join(path, cls))


if __name__ == "__main__":
    # delete(train_path)
    # delete(val_path)
    path = r'C:\Users\hkuit164\Desktop\sport_classification\data\src\video\golf0210_skeleton\standing\frame'
    Utils.delete_wrong(path)