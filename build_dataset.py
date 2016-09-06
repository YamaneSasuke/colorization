# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 17:59:14 2016

@author: yamane
"""

import numpy as np
from skimage import color, io
import glob
import os


def rgb_hwc2hsv_chw(image):
    hsv_hwc = color.rgb2hsv(image)
    hsv_chw = np.transpose(hsv_hwc, (2, 0, 1))
    return hsv_chw


if __name__ == '__main__':
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','t','u','v','w','y']
    f = open("file_list.txt", "w")
    for i in range(23):
        path = r"C:\Users\yamane\Desktop\dataset\data\vision\torralba\deeplearning\images256"
        path = os.path.join(path, str(alphabet[i]))
        label_list = glob.glob(os.path.join(path, '*'))
        for label in label_list:
            image_list = glob.glob(os.path.join(label, '*.jpg'))
            for index, filename in enumerate(image_list):
                image = io.imread(filename)
                if len(image.shape) == 2:
                    continue
                hsv_chw = rgb_hwc2hsv_chw(image)
                s = hsv_chw[1]
                if s.std() >= 0.14:
                    f.write(filename + "\n")
    f.close()
