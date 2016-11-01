# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:43:57 2016

@author: yamane
"""

import os
import sys
import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from skimage import io, color, transform
import tqdm


def rgb_hwc2hsv_chw(image):
    hsv_hwc = color.rgb2hsv(image)
    hsv_chw = np.transpose(hsv_hwc, (2, 0, 1))
    return hsv_chw


def create_path_list(data_location, dataset_root_dir, threshold):
    path_list = []
    root_dir_path = os.path.join(data_location, dataset_root_dir)

    for root, dirs, files in os.walk(root_dir_path):
        for file_name in tqdm.tqdm(files):
            file_path = os.path.join(root, file_name)
            image = io.imread(file_path)
            if len(image.shape) == 2:
                continue
            hsv_chw = rgb_hwc2hsv_chw(image)
            s = hsv_chw[1]
            if s.std() >= threshold:
                path_list.append(file_path)
    return path_list


def output_path_list(path_list, output_root_dir):
    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.txt'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = open(output_root_dir, "w")
    for path in path_list:
        f.write(path + "\n")
    f.close()


def output_hdf5(path_list, data_chw, output_root_dir):
    num_data = len(path_list)
    class_list = []

    channel, height, width = data_chw

    dirs = output_root_dir.split('\\')
    file_name = dirs[-1] + '.hdf5'
    output_root_dir = os.path.join(output_root_dir, file_name)

    f = h5py.File(output_root_dir, mode='w')
    image_features = f.create_dataset('image_features',
                                      (num_data, channel, height, width),
                                      dtype='uint8')
    targets = f.create_dataset('targets', (num_data,), dtype='uint8')

    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'channel'
    image_features.dims[2].label = 'height'
    image_features.dims[3].label = 'width'
    targets.dims[0].label = 'batch'

    for path in path_list:
        path = path.strip()
        dirs = path.split('\\')
        images256_index = dirs.index('images256')
        class_list.append('_'.join(dirs[images256_index+2:-1]))
    class_uniq = list(set(class_list))

    try:
        for i in tqdm.tqdm(range(num_data)):
            image = io.imread(path_list[i])
            image = transform.resize(image, (height, width))
            image = np.transpose(image, (2, 0, 1))
            image = np.reshape(image, (1, channel, height, width))
            image = image * 256
            image_features[i] = image
            targets[i] = class_uniq.index(class_list[i])

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    f.flush()
    f.close()


def main(data_location, output_location, output_size, test_size):
    dataset_root_dir = r'data\vision\torralba\deeplearning\images256'
    output_dir_name = 'raw_dataset_' + str(output_size)
    output_root_dir = os.path.join(output_location, output_dir_name)
    data_chw = (3, output_size, output_size)
    threshold = 0.20

    if os.path.exists(output_root_dir):
        print u"すでに存在するため終了します."
        sys.exit()
    else:
        os.makedirs(output_root_dir)

    path_list = create_path_list(data_location, dataset_root_dir, threshold)
    shuffled_path_list = np.random.permutation(path_list)
    output_path_list(shuffled_path_list, output_root_dir)
    output_hdf5(shuffled_path_list, data_chw, output_root_dir)


if __name__ == '__main__':
    data_location = r'E:'
    output_location = r'E:\raw_dataset'
    output_size = 56

    main(data_location, output_location, output_size)
