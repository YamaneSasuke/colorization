# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 17:59:14 2016

@author: yamane
"""

import os
import numpy as np
from skimage import color, io


def rgb_hwc2hsv_chw(image):
    hsv_hwc = color.rgb2hsv(image)
    hsv_chw = np.transpose(hsv_hwc, (2, 0, 1))
    return hsv_chw


if __name__ == '__main__':
    data_location = r'C:\Users\yamane\Desktop\dataset'
    dataset_root_dir = r'data\vision\torralba\deeplearning\images256'
    root_dir_path = os.path.join(data_location, dataset_root_dir)
    f = open("file_list.txt", "w")
    for root, dirs, files in os.walk(root_dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            image = io.imread(file_path)
            if len(image.shape) == 2:
                continue
            hsv_chw = rgb_hwc2hsv_chw(image)
            s = hsv_chw[1]
            if s.std() >= 0.20:
                f.write(file_path + "\n")
    f.close()
