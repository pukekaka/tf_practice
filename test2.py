import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import time


num_puntos = 1000
conjunto_puntos = []
for i in range(num_puntos):
    x1 = np.random.normal(0.0, 0.5)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    conjunto_puntos.append([x1, y1])
x_data = [v[0] for v in conjunto_puntos]
y_data = [v[1] for v in conjunto_puntos]

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# dw, db = tf.gradients(loss, [w, b])
# update_w = tf.assign(w, w - 0.5 * dw)
# update_b = tf.assign(b, b - 0.5 * db)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(init)
    for step in range(16):
        print("step " + str(step))
        sess.run(train)
        # sess.run([update_w, update_b])
        plt.plot(x_data, y_data, 'ro', label='train data')
        plt.plot(x_data, sess.run(w) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-2, 2)
        plt.legend()
        # time.sleep(1)
        plt.show()

