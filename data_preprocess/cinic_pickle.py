# -*- coding: utf-8 -*-
import pickle

import numpy as np
import torchvision
from PIL import Image

if __name__ == '__main__':
    # dtypes = ['train', 'test']
    dtypes = ['train']
    data_folder="../../data/CINIC-L/"
    for dtype in dtypes:
        output = open('{}{}.pkl'.format(data_folder, dtype), 'wb')
        entry = dict()
        entry_data = None
        image_folder = torchvision.datasets.ImageFolder(root=data_folder + dtype)
        image_paths = image_folder.imgs
        for (path, label) in image_paths:
            img = Image.open(path).convert("RGB").resize((32, 32))
            temp = np.expand_dims(np.asarray(img), axis=0)
            if entry_data is None:
                entry_data = temp
            else:
                entry_data = np.concatenate((entry_data, temp), axis=0)
        entry['data'] = entry_data
        entry['labels'] = image_folder.targets
        pickle.dump(entry, output)
