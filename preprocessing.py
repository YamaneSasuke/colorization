# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 16:14:41 2016

@author: yamane
"""

import numpy as np
from skimage import io, transform
import os

if __name__ == '__main__':
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','t','u','v','w','y']
    f = open("file_list.txt", "r")
    i = 0
    for path in f:
        path = path.strip()
        image_hwc_nomal = io.imread(path)
        image_hwc = transform.resize(image_hwc_nomal, (224, 224))
        root = r'\Users\yamane\Desktop\new_dataset'
        initial = path.split('\\')[10: -1]
        new_path = os.path.join(root, *initial)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        file_name = path.split('\\')[-1]
        save_path = os.path.join(new_path, file_name)
        io.imsave(save_path, image_hwc, quality=100)
        i = i + 1
    f.close()
