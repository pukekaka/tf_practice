import tensorflow as tf
import numpy as np

from practice.mann.helper.utils import shared_float32
from practice.mann.helper.init import shared_one_hot

def memory_augmented_neural_network(batch_size=16, memory_shape=(128,40),\
                                    controller_size=200, nb_reads=4):

    M_0 = shared_float32(1e-6 * np.ones((batch_size,) + memory_shape), name='memory')
    c_0 = shared_float32(np.zeros((batch_size, controller_size)), name='memory_cell_state')
    h_0 = shared_float32(np.zeros((batch_size, controller_size)), name='hidden_state')
    r_0 = shared_float32(np.zeros((batch_size, nb_reads * memory_shape[1])), name='read_vector')
    wr_0 = shared_one_hot((batch_size, nb_reads, memory_shape[0]), name='wr')
    wu_0 = shared_one_hot((batch_size, memory_shape[0]), name='wu')
