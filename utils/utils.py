# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import cv2
import time
from PIL import Image
import numpy as np
import heapq
import os
from config import config
import subprocess


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def delete_wrong(path):
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            if os.path.getsize(img_path) < 100:
                os.remove(img_path)

    @staticmethod
    def cv2_to_pillow(cv2_array):
        pillow_image_object = Image.fromarray(cv2.cvtColor(cv2_array, cv2.COLOR_BGR2RGB))
        return pillow_image_object

    @staticmethod
    def pillow_to_cv2(pillow_array):
        cv2_image_object = cv2.cvtColor(np.array(pillow_array), cv2.COLOR_RGB2BGR)
        return cv2_image_object

    @staticmethod
    def image_sharp(image_object):
        gray_lap = cv2.Laplacian(image_object, cv2.CV_16S, ksize=3)
        dst = cv2.convertScaleAbs(gray_lap)
        im_mean, im_std = cv2.meanStdDev(dst)
        return im_mean[0] * im_std[0]

    '''
    选取关键帧
    '''
    @staticmethod
    def get_max_frame(video_path):
        video_reader = cv2.VideoCapture()
        video_reader.open(video_path)
        frame_list = []
        frame_count = 0
        flag, video_frame = video_reader.read()
        frame_list.append(video_frame)
        while flag:
            if frame_count % 10 == 0:
                frame_list.append(video_frame)
            flag, video_frame = video_reader.read()
            frame_count += 1

        try:
            max_sharp = -sys.maxsize
            max_sharp_frame = frame_list[0]
            for im_frame in frame_list:
                im_sharp = Utils.image_sharp(im_frame)
                if im_sharp > max_sharp:
                    max_sharp_frame = im_frame
                    max_sharp = im_sharp
            return max_sharp_frame
        except:
            return frame_list[0]

    @staticmethod
    def detect_rotate(path):
        cmd = r'../ffprobe.exe ' + path
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if b'rotate          : 270' in err:
            return 270
        elif b'rotate          : 90' in err:
            return 90
        elif b'rotate          : 180' in err:
            return 180
        else:
            return 0

    @staticmethod
    def image_rotate(img, flag):
        if flag == 270:
            img = np.rot90(img)
            return img
        elif flag == 180:
            img = np.rot90(img)
            img = np.rot90(img)
            return img
        elif flag == 90:
            img = np.rot90(img)
            img = np.rot90(img)
            img = np.rot90(img)
            return img
        elif flag == None:
            return img
        else:
            print("A wrong image input")

    @staticmethod
    def get_top_5_frame_bak(video_path, rorated=False):
        video_reader = cv2.VideoCapture()
        video_reader.open(video_path)
        frame_list = []
        frame_count = 0
        flag, video_frame = video_reader.read()
        while flag:
            if frame_count % 5 == 0:
                frame_list.append(video_frame)
            flag, video_frame = video_reader.read()
            frame_count += 1
        if len(frame_list) <= 5:
            return frame_list
        else:
            try:
                frame_score_list = []
                for idx, frame in enumerate(frame_list):
                    # print(frame)
                    frame_score = Utils.image_sharp(frame)
                    frame_score_list.append(frame_score)

                score_index_list = [[index, score] for index, score in enumerate(frame_score_list)]
                score_index_list = sorted(score_index_list, key=lambda x: x[1], reverse=True)

                result_frame_list = []
                for index, score in score_index_list[:5]:
                    result_frame_list.append(frame_list[index])
                return result_frame_list

            except Exception as e:
                print("frame error:", e)
                return frame_list[:5]

    @staticmethod
    def image_rotate(image_object):
        new_image = np.rot90(image_object)
        new_image = np.rot90(new_image)
        new_image = np.rot90(new_image)
        return new_image

    @staticmethod
    def align(img, box, output_size, ratio):
        src = np.array([
            [0, 0],
            [112, 0],
            [0, 112],
        ], dtype=np.float32)
        center_x = (output_size - 1) / 2
        center_y = (output_size - 1) / 2
        src = src * output_size / 112.0
        src[:, 0] = (src[:, 0] - center_x) / 100 * ratio + center_x
        src[:, 1] = (src[:, 1] - center_y) / 100 * ratio + center_y  # for image size = 112
        rectangle = Utils.rectangledo(box)
        dst = rectangle.astype(np.float32)
        M = cv2.estimateRigidTransform(dst.reshape(1, 3, 2), src.reshape(1, 3, 2), False)
        warped = cv2.warpAffine(img, M, (output_size, output_size), borderValue=0.0)
        return warped

    @staticmethod
    def rectangledo(box):
        img_point = []
        x = float(box[0])
        y = float(box[1])
        w = float(box[2])
        h = float(box[3])
        y_offset = (h - w) / 2
        img_point.append([x, y + y_offset])
        img_point.append([x + w, y + y_offset])
        img_point.append([x, y + h - y_offset])
        return np.array(img_point)

    @staticmethod
    def get_top_5_frame(video_path, rotated=False):
        video_reader = cv2.VideoCapture()
        video_reader.open(video_path)
        frame_list = []
        legal_flag = 0
        flag, video_frame = video_reader.read()
        frame_count = 0
        while flag:
            if frame_count % 5 == 0:
                frame_list.append(video_frame)
            flag, video_frame = video_reader.read()
            frame_count += 1
        # print(frame_count)
        if frame_count < 29:
            legal_flag = -1

        if rotated:
            for i in range(len(frame_list)):
                frame_list[i] = Utils.image_rotate(frame_list[i])

        # print(len(frame_list))

        if len(frame_list) <= 5:
            return frame_list, legal_flag

        else:
            try:
                frame_score_list = []
                for idx, frame in enumerate(frame_list):
                    # print(frame)
                    frame_score = Utils.image_sharp(frame)
                    frame_score_list.append(frame_score)

                score_index_list = [[index, score] for index, score in enumerate(frame_score_list)]
                score_index_list = sorted(score_index_list, key=lambda x: x[1], reverse=True)

                result_frame_list = []
                for index, score in score_index_list[:5]:
                    result_frame_list.append(frame_list[index])
                return result_frame_list, legal_flag

            except Exception as e:
                print("frame error:", e)
                return frame_list[:5], legal_flag
            
    @staticmethod
    def check_image_score(face_score, whole_score, attack_score=-1.0):
        if attack_score >= attack_threshold:
            return False

        if abs(whole_score - face_score) <= config.diff_threshold:
            if whole_score >= whole_threshold and face_score >= face_threshold:
                return True
            else:
                return False
        else:
            avg_score = config.whole_weight*whole_score + (1.0-config.whole_weight)*face_score
            if avg_score >= avg_threshold:
                return True
            else:
                return False

    @staticmethod
    def calculate_normal_score(score_list, normal_tag_list):
        normal_score = 0.0
        for tag in normal_tag_list:
            normal_score += score_list[tag]
        return normal_score

    # 基于人脸区域进行图片scale
    @staticmethod
    def scale_image(face_x1, face_x2, face_y1, face_y2, image_array, scale_num=0.3):
        origin_img_shape = image_array.shape

        x_len = abs(face_x1 - face_x2)
        y_len = abs(face_y1 - face_y2)

        scale_x_len = int(x_len * scale_num)
        scale_y_len = int(y_len * scale_num)

        scale_x1 = max(0, face_x1 - scale_x_len)
        scale_x2 = min(origin_img_shape[1], face_x2 + scale_x_len)
        scale_y1 = max(0, face_y1 - scale_y_len)
        scale_y2 = min(origin_img_shape[1], face_y2 + scale_y_len)

        scaled_image = np.asarray(image_array[scale_y1:scale_y2, scale_x1:scale_x2])

        return scaled_image


if __name__ == "__main__":
    video_path = r"D:\projects\silent_liveness_detection\data\clean_test_data\attack_video\attack_roate\attack_39.MOV"
    result_frame_list, video_flag = Utils.get_top_5_frame(video_path, True)
    result_dir = r"D:\projects\silent_liveness_detection\data\temp"
    print(len(result_frame_list))
    for idx, frame in enumerate(result_frame_list):
        cv2.imwrite(os.path.join(result_dir,"%d.jpg"% idx), frame)
    print(video_flag)

