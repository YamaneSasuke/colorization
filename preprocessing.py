# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 16:14:41 2016

@author: yamane
"""

import os
from skimage import io, transform
import numpy as np


def build_resized_dataset(output_size=224, data_location='', quality=100):
    output_dir_name = 'resized_dataset_' + str(output_size)
    output_root_dir = os.path.join(data_location, output_dir_name)
    if os.path.exists(output_root_dir):
        print u"すでに存在するため終了します."
        return
    else:
        os.makedirs(output_root_dir)

    f_file = open("file_list.txt", "r")
    f_path = open("random_file_path.txt", "w")
    paths = []
    for path in f_file:
        paths.append(path)
    paths = np.array(paths)

    random_paths = np.random.permutation(paths)

    for path in random_paths:
        path = path.strip()
        dirs = path.split('\\')
        file_name = dirs[-1]
        images256_index = dirs.index('images256')

        image = io.imread(path)
        image_resized = transform.resize(image, (output_size, output_size))
        sub_dirs = dirs[images256_index+1: -1]
        output_dir_path = os.path.join(output_root_dir, *sub_dirs)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        save_path = os.path.join(output_dir_path, file_name)
        io.imsave(save_path, image_resized, quality=quality)
        f_path.write(path + "\n")
    f_file.close()
    f_path.close()

if __name__ == '__main__':
    output_size = 56
    data_location = r'\Users\yamane\Desktop\dataset'
    quality = 100

    build_resized_dataset(output_size, data_location, quality)
