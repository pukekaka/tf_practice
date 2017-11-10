import tensorflow as tf
import numpy as np
from practice.mann.helper import generator as gn
from practice.mann.helper.utils import shared_float32
from practice.mann.helper.init import shared_one_hot


nb_reads = 4
controller_size = 200
memory_shape = (128, 40)
nb_class = 5
input_size = 20 * 20
batch_size = 16
nb_samples_per_class = 10

# data load
generator = gn.OmniglotGenerator(data_folder='./data/omniglot',
                                 batch_size=batch_size,
                                 nb_samples=nb_class,
                                 nb_samples_per_class=nb_samples_per_class,
                                 max_rotation=0.,
                                 max_shift=0.,
                                 max_iter=None)


# memory bank
M_0 = shared_float32(1e-6 * np.ones((batch_size,) + memory_shape), name='memory')
c_0 = shared_float32(np.zeros((batch_size, controller_size)), name='memory_cell_state')
h_0 = shared_float32(np.zeros((batch_size, controller_size)), name='hidden_state')
r_0 = shared_float32(np.zeros((batch_size, nb_reads * memory_shape[1])), name='read_vector')
wr_0 = shared_one_hot((batch_size, nb_reads, memory_shape[0]), name='wr')
wu_0 = shared_one_hot((batch_size, memory_shape[0]), name='wu')
memory_bank = tf.sacn(initializer=[M_0, c_0, h_0, r_0, wr_0, wu_0])

# controller
def shape_high(shape):
    shape = np.array(shape)
    if isinstance(shape, int):
        high = np.sqrt(6. / shape)
    else:
        high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
    return (shape, high)


with tf.variable_scope("Weights"):
    shape, high = shape_high((nb_reads, controller_size, memory_shape[1]))
    W_key = tf.get_variable('W_key', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    b_key = tf.get_variable('b_key', shape=(nb_reads, memory_shape[1]), initializer=tf.constant_initializer(0))
    shape, high = shape_high((nb_reads, controller_size, memory_shape[1]))
    W_add = tf.get_variable('W_add', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    b_add = tf.get_variable('b_add', shape=(nb_reads, memory_shape[1]), initializer=tf.constant_initializer(0))
    shape, high = shape_high((nb_reads, controller_size, 1))
    W_sigma = tf.get_variable('W_sigma', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    b_sigma = tf.get_variable('b_sigma', shape=(nb_reads, 1), initializer=tf.constant_initializer(0))
    shape, high = shape_high((input_size + nb_class, 4 * controller_size))
    W_xh = tf.get_variable('W_xh', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    b_h = tf.get_variable('b_xh', shape=(4 * controller_size), initializer=tf.constant_initializer(0))
    shape, high = shape_high((controller_size + nb_reads * memory_shape[1], nb_class))
    W_o = tf.get_variable('W_o', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    b_o = tf.get_variable('b_o', shape=(nb_class), initializer=tf.constant_initializer(0))
    shape, high = shape_high((nb_reads * memory_shape[1], 4 * controller_size))
    W_rh = tf.get_variable('W_rh', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    shape, high = shape_high((controller_size, 4 * controller_size))
    W_hh = tf.get_variable('W_hh', shape=shape, initializer=tf.random_uniform_initializer(-1 * high, high))
    gamma = tf.get_variable('gamma', shape=[1], initializer=tf.constant_initializer(0.95))

def step(x):
    last_read = memory_bank * weight_vector