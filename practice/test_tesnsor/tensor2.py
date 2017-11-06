import tensorflow as tf

hello = tf.constant('hello, tensorflow!')
print(hello)

a = tf.constant(10)
b = tf.constant(4)
c = tf.add(a,b)
print(c)

session = tf.Session()

print(session.run(hello))
print(session.run(c))

X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1,2,3], [4,5,6]]

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

expr = tf.matmul(X, W) + b
session.run(tf.global_variables_initializer())

print(session.run(W))
print(session.run(b))
print('expr', session.run(expr, feed_dict={X: x_data}))

session.close()
