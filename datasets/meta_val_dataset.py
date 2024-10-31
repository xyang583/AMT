import os
import h5py
from PIL import Image
import numpy as np

import torch


class MetaValDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, num_episodes=4000):
        super().__init__()

        self.num_episodes = num_episodes
        self.h5_path = h5_path
        self.h5_file = None

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        record = self.h5_file[str(idx)]
        support_images = record['sx'][()]
        support_labels = record['sy'][()]
        query_images = record['x'][()]
        query_labels = record['y'][()]


        sorted_idx = np.argsort(support_labels)
        support_images = support_images[sorted_idx]
        support_labels = support_labels[sorted_idx]

        sorted_idx = np.argsort(query_labels)
        query_images = query_images[sorted_idx]
        query_labels = query_labels[sorted_idx]

        return support_images, support_labels, query_images, query_labels