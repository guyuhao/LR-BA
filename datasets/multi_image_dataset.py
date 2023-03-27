# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Tuple

import cv2
import torch
from torchvision.datasets import VisionDataset

from datasets.common import add_pixel_pattern_backdoor

"""
define multiple image dataset, support multiple parties, used for BHI
"""

class MultiImageDataset(VisionDataset):
    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,
            party_num=2
    ) -> None:

        super(MultiImageDataset, self).__init__(root="", transform=transform,
                                                target_transform=target_transform)
        self.data: Any = []  # X
        self.targets = []  # Y

        self.data = X
        self.targets = y

        self.backdoor_indices = backdoor_indices  # backdoor indices of dataset
        self.party_num = party_num  # parties number

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_groups, target = self.data[index], self.targets[index]
        images_list = []
        # split images into parties
        for img_id in range(self.party_num):
            img_path = img_groups[img_id]
            image = img_path
            if self.transform is not None:
                image = self.transform(image)
            # add trigger if index is in backdoor indices, only for the first passive party
            if img_id == 1:
                if self.backdoor_indices is not None and index in self.backdoor_indices:
                    image = add_pixel_pattern_backdoor(image)

            images_list.append(image)
        images = torch.stack(tuple(image for image in images_list), 0)
        return images, target

    def __len__(self) -> int:
        return len(self.data)
