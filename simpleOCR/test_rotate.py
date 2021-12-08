import re
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from utils import rotate_box,align_box,get_idx
from tect_detect.test_text_detect import test_net,net,refine_net,poly
from PIL import Image
import torch

import matplotlib.pyplot as plt
from PIL import Image

import joblib


def test(image):
    image_copy = image.copy()
    bboxes, polys, score_text = test_net(net, image_copy, 0.7, 0.4, 0.4, True, poly, refine_net)
    if bboxes != []:
        bboxes_xxyy = []
        ratios = []
        degrees = []
        for box in bboxes:
            x_min = min(box, key=lambda x: x[0])[0]
            x_max = max(box, key=lambda x: x[0])[0]
            y_min = min(box, key=lambda x: x[1])[1]
            y_max = max(box, key=lambda x: x[1])[1]
            if (x_max - x_min) > 20:
                ratio = (y_max - y_min) / (x_max - x_min)
                ratios.append(ratio)

        mean_ratio = np.mean(ratios)
        if mean_ratio >= 1:
            image, bboxes = rotate_box(image, bboxes, None, True, False)

        bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, poly, refine_net)

        image, check = align_box(image, bboxes, skew_threshold=0.9)

        if check:
            bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, poly, refine_net)
        h, w, c = image.shape

        for box in bboxes:
            x_min = max(int(min(box, key=lambda x: x[0])[0]), 1)
            x_max = min(int(max(box, key=lambda x: x[0])[0]), w - 1)
            y_min = max(int(min(box, key=lambda x: x[1])[1]), 3)
            y_max = min(int(max(box, key=lambda x: x[1])[1]), h - 2)
            bboxes_xxyy.append([x_min - 1, x_max, y_min - 1, y_max])

        img_copy = image.copy()
        for b in bboxes_xxyy:
            cv2.rectangle(img_copy, (b[0], b[2]), (b[1], b[3]), (255, 0, 0), 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_copy)
        return img_copy


if __name__ == "__main__":
    path = 'result/res_img1.jpg'
    image = cv2.imread(path)
    test_image = test(image)
    cv2.imwrite('test.jpg', test_image)

