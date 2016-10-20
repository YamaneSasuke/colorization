# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:09:53 2016

@author: yamane
"""

import os

if __name__ == '__main__':
    data_location = r'C:\Users\yamane\Desktop\dataset'
    dataset_root_dir = r'resized_dataset_56'
    root_dir_path = os.path.join(data_location, dataset_root_dir)
    f = open("image_path_list.txt", "w")
    for root, dirs, files in os.walk(root_dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            f.write(file_path + "\n")
    f.close()
