import tensorflow as tf
import numpy as np
from practice.mann.helper import generator as gn

def omniglot():

    sess = tf.InteractiveSession()

    input_ph = tf.placeholder(dtype=tf.float32, shape=(16, 50, 400)) #(batch_size, time, input_dim)
    target_ph = tf.placeholder(dtype=tf.int32, shape=(16, 50)) # (batch_size, time)

    nb_reads = 4
    controller_size = 200
    memory_shape = (128,40)
    nb_class = 5
    input_size = 20 * 20
    batch_size = 16
    nb_samples_per_class = 10

    generator = gn.OmniglotGenerator(data_folder='./data/omniglot',
                                  batch_size=batch_size,
                                  nb_samples=nb_class,
                                  nb_samples_per_class=nb_samples_per_class,
                                  max_rotation=0.,
                                  max_shift=0.,
                                  max_iter=None)

    it = iter(generator)

    f = open("test", 'w')
    while True:
        try:
            e = next(it)
            f.write(str(e))
        except StopIteration:
            break
    f.close()
    print('end')

if __name__ == '__main__':
    omniglot()