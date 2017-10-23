import sys, os
sys.path.append(os.pardir)
from practice.cnn import input_data
import tensorflow as tf


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add\
                          (tf.nn.conv2d(img, w, \
                                        strides=[1, 1, 1, 1], \
                                        padding='SAME'), b))


def max_pool(img, k):
    return tf.nn.max_pool(img, \
                          ksize=[1, k, k, 1],\
                          strides=[1, k, k, 1],\
                          padding="VALID")

mnist = input_data.read_data_sets(os.getcwd(), one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 784
n_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

_X = tf.reshape(x, shape=[-1, 28, 28, 1])

wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
bc1 = tf.Variable(tf.random_normal([32]))
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))
wout = tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
bout = tf.Variable(tf.random_normal([n_classes]))

conv1 = conv2d(_X, wc1, bc1)
conv1 = max_pool(conv1, k=2)

keep_prob = tf.placeholder(tf.float32)
conv1 = tf.nn.dropout(conv1, keep_prob)

conv2 = conv2d(conv1, wc2, bc2)
conv2 = max_pool(conv2, k=2)
conv2 = tf.nn.dropout(conv2, keep_prob)

dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1)) # Relu activation
dense1 = tf.nn.dropout(dense1, keep_prob) # Apply Dropout

# Output, class prediction
pred = tf.add(tf.matmul(dense1, wout), bout)

#pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
