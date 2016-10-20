# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:43:57 2016

@author: yamane
"""

import numpy as np
from skimage import io
import tqdm
import h5py
from fuel.datasets.hdf5 import H5PYDataset


if __name__ == '__main__':
    # 超パラメータ
    channel = 3
    height = 56
    width = 56
    test_size = 20000

    image_list = []
    images = []
    f = open(r"C:\Users\yamane\Dropbox\colorization\random_file_path.txt", "r")
    for path in f:
        path = path.strip()
        image_list.append(path)
    f.close()

    num_images = len(image_list)
    num_train = num_images - test_size

    f = h5py.File(r'E:\raw_resized_dataset_random_save_56\raw_dataset_56.hdf5',
                  mode='w')

    image_features = f.create_dataset('image_features',
                                      (num_images, channel, height, width),
                                      dtype='uint8')
    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'channel'
    image_features.dims[2].label = 'height'
    image_features.dims[3].label = 'width'

    try:
        for i in tqdm.tqdm(range(num_images)):
            image = io.imread(image_list[i])
            image = np.transpose(image, (2, 0, 1))
            image = np.reshape(image, (1, channel, height, width))
            image_features[i] = image

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    split_dict = {
            'train': {'image_features': (0, num_train)},
            'test': {'image_features': (num_train, num_images)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
