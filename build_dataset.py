# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 17:59:14 2016

@author: yamane
"""

import numpy as np
from skimage import color, io
import glob


def rgb_hwc2hsv_chw(image):
    hsv_hwc = color.rgb2hsv(image)
    hsv_chw = np.transpose(hsv_hwc, (2, 0, 1))
    return hsv_chw


if __name__ == '__main__':
    image_list = glob.glob('./dataset/*.jpg')
    for i, filename in enumerate(image_list):
        image = io.imread(filename)

        if len(image.shape) == 2:
            continue

        hsv_chw = rgb_hwc2hsv_chw(image)
        s = hsv_chw[1]
        if s.std() >= 0.14:
            io.imsave('./color\\' + str(i) + '.jpg', image)
