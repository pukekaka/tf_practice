import numpy as np
import os
import random

class OmniglotGenerator(object):

    def __init__(self, data_folder, batch_size=1, nb_samples=5, nb_samples_per_class=10, max_rotation=-np.pi/6, max_shift=10, img_size=(20,20), max_iter=None):
        super(OmniglotGenerator, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.nb_samples = nb_samples