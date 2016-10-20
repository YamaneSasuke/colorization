# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:43:57 2016

@author: yamane
"""

import numpy as np
from skimage import io
import tqdm
import h5py


if __name__ == '__main__':
    # 超パラメータ
    channel = 3
    height = 56
    width = 56

    image_list = []
    images = []
    f = open(r"hdd\random_file_path.txt", "r")
    for path in f:
        path = path.strip()
        image_list.append(path)
    f.close()

    num_images = len(image_list)

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

    f.flush()
    f.close()
