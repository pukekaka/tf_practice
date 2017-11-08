import numpy as np

import os
import random

class OmniglotGenerator(object):

    def __init__(self,
                 data_folder,
                 batch_size=1,
                 nb_samples=5,
                 nb_samples_per_class=10,
                 max_rotation=-np.pi/6,
                 max_shift=10,
                 img_size=(20,20),
                 max_iter=None):
        super(OmniglotGenerator, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.nb_samples = nb_samples
        self.nb_samples_per_class = nb_samples_per_class
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.character_folders = [os.path.join(self.data_folder,
                                               family,
                                               character) \
                                  for family in os.listdir(self.data_folder) \
                                  if os.path.isdir(os.path.join(self.data_folder,
                                                                family)) \
                                  for character in os.listdir(os.path.join(self.data_folder,
                                                                           family))]
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample(self.nb_samples)