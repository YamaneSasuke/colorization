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
    for i in range(23):
        path = '/Users/yamane/Desktop/dataset/data/vision/torralba/deeplearning/images256/'
        path = path + str(alphabet[i]) + '/'
        label_list = glob.glob(str(path) + '*')
        for label in label_list:
            new_path = '/Users/yamane/Desktop/new_dataset/' + str(alphabet[i]) + '/' + str(label.split('\\')[-1])
            os.makedirs(new_path)
            image_list = glob.glob(str(label) + '/*.jpg')
            for index, filename in enumerate(image_list):
                image = io.imread(filename)

                if len(image.shape) == 2:
                    continue

                hsv_chw = rgb_hwc2hsv_chw(image)
                s = hsv_chw[1]
                if s.std() >= 0.14:
                    io.imsave(new_path + '/' + str(filename.split('\\')[-1]) + '.jpg', image)
