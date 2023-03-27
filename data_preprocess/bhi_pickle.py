# -*- coding: utf-8 -*-
import pickle
from glob import glob

import cv2
import numpy as np
import torchvision
from PIL import Image

if __name__ == '__main__':
    data_folder="../../data/BHI/BC_IDC_muscle/"
    output = open('{}data.pkl'.format(data_folder), 'wb')
    entry = dict()
    entry_data = None
    img_paths_list = glob(data_folder + '/**/*.png', recursive=True)
    labels = []
    for i, img_path in enumerate(img_paths_list):
        label = int(img_path[-5])
        img = cv2.imread(img_path)
        image = cv2.resize(img, (50, 50))
        temp = np.expand_dims(np.asarray(image), axis=0)
        if entry_data is None:
            entry_data = temp
        else:
            entry_data = np.concatenate((entry_data, temp), axis=0)
        labels.append(label)
    entry['data'] = entry_data
    entry['labels'] = np.array(labels)
    pickle.dump(entry, output)
