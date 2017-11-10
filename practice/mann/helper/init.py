import tensorflow as tf
import numpy as np
import sys


def shared_one_hot(shape, dtype=tf.float32, name='', n=None):
	shape = (shape,) if isinstance(shape,int) else shape
	shape = shape if n is None else (n,) + shape
	initial_vector = np.zeros(shape, dtype=np.float32)
	initial_vector[...,0] = 1
	return tf.Variable(tf.cast(initial_vector, tf.float32), name=name)