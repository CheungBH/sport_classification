import os
import cv2

step = 3
main_folder = '../data/src/video/golf0210_skeleton'
selected_class = "downswing"


class VideoProcessor(object):
    def __init__(self, path):
        self.video_folder = path
        self.videos = os.listdir(self.video_folder)
        self.step = step

    def process(self):
        dest_folder = os.path.join(self.video_folder, "frame")
        os.makedirs(dest_folder, exist_ok=True)
        for video in self.videos:
            video_path = os.path.join(self.video_folder, video)
            cap = cv2.VideoCapture(video_path)
            cnt = 0
            while True and not cv2.waitKey(1) == 27:
                ret_val, self.img = cap.read()
                if ret_val:
                    cnt += 1
                    ret, frame = cap.read()
                    if cnt % self.step == 0:
                        cv2.imwrite(os.path.join(dest_folder, video[:-4]) + "_{}.jpg".format(cnt), frame)
                else:
                    break


if __name__ == '__main__':
    for action in os.listdir(main_folder):
        if action == selected_class and selected_class:
            VP = VideoProcessor(os.path.join(main_folder, action))
            VP.process()
#     path = "../../Datas/ice_hockey"
#     VP = VideoProcessor(path)
#     VP.process()