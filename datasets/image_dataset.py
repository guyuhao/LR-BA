# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from datasets.common import add_pixel_pattern_backdoor

"""
define image dataset, only support two parties, used for CIFAR and CINIC
"""


class ImageDataset(VisionDataset):

    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,
            half=None
    ) -> None:

        super(ImageDataset, self).__init__(root="", transform=transform,
                                           target_transform=target_transform)
        self.data: Any = []  # X
        self.targets = []  # Y

        self.data = X
        self.targets = y

        self.backdoor_indices = backdoor_indices  # backdoor indices of dataset

        self.half = half  # vertical halves to split

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any], Any]:
        img, target = self.data[index], self.targets[index]
        if type(img) is np.str_:
            img = Image.open(img)
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # split image into halves vertically for parties
        img_a, img_b = img[:, :, :self.half], img[:, :, self.half:]

        if self.target_transform is not None:
            target = self.target_transform(target)

        # add trigger if index is in backdoor indices
        if self.backdoor_indices is not None and index in self.backdoor_indices:
            img_b = add_pixel_pattern_backdoor(img_b)

        return (img_a, img_b), target

    def __len__(self) -> int:
        return len(self.data)
