import numpy as np
import tensorflow as tf
import util.mann_cell as mann_cell


class MANN():
    def __init__(self, values):
        if values.label_type == 'one_hot':
            values.output_dim = values.n_classes

        elif values.label_type == 'five_hot':
            values.output_dim = 25

        self.x_image = tf.placeholder(dtype=tf.float32, shape=[values.batch_size, values.seq_length, values.image_width * values.image_height])
        self.x_label = tf.placeholder(dtype=tf.float32, shape=[values.batch_size, values.seq_length, values.output_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[values.batch_size, values.seq_length, values.output_dim])

        if values.model == 'MANN':
            cell = mann_cell.MANNCell(values.rnn_size, values.memory_size, values.memory_vector_dim, head_num=values.read_head_num)
        # elif args.model == 'MANN2':
        #     import practice.rnn.notuse.mann_cell_2 as mann_cell
        #     cell = mann_cell.MANNCell(args.rnn_size, args.memory_size, args.memory_vector_dim,
        #                             head_num=args.read_head_num)

        state = cell.zero_state(values.batch_size, tf.float32)
        self.state_list = [state]   # For debugging
        self.o = []
        for t in range(values.seq_length):
            output, state = cell(tf.concat([self.x_image[:, t, :], self.x_label[:, t, :]], axis=1), state)
            # output, state = cell(self.y[:, t, :], state)
            with tf.variable_scope("o2o", reuse=(t > 0)):
                o2o_w = tf.get_variable('o2o_w', [output.get_shape()[1], values.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                o2o_b = tf.get_variable('o2o_b', [values.output_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                                        # initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
                output = tf.nn.xw_plus_b(output, o2o_w, o2o_b)

            if values.label_type == 'one_hot':
                output = tf.nn.softmax(output, dim=1)

            elif values.label_type == 'five_hot':
                output = tf.stack([tf.nn.softmax(o) for o in tf.split(output, 5, axis=1)], axis=1)
            self.o.append(output)
            self.state_list.append(state)
        self.o = tf.stack(self.o, axis=1)
        self.state_list.append(state)

        eps = 1e-8
        if values.label_type == 'one_hot':
            self.learning_loss = -tf.reduce_mean(  # cross entropy function
                tf.reduce_sum(self.y * tf.log(self.o + eps), axis=[1, 2])
            )
        elif values.label_type == 'five_hot':
            self.learning_loss = -tf.reduce_mean(  # cross entropy function
                tf.reduce_sum(tf.stack(tf.split(self.y, 5, axis=2), axis=2) * tf.log(self.o + eps), axis=[1, 2, 3])
            )
        self.o = tf.reshape(self.o, shape=[values.batch_size, values.seq_length, -1])
        self.learning_loss_summary = tf.summary.scalar('learning_loss', self.learning_loss)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=values.learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(
            #     learning_rate=args.learning_rate, momentum=0.9, decay=0.95
            # )
            # gvs = self.optimizer.compute_gradients(self.learning_loss)
            # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            # self.train_op = self.optimizer.apply_gradients(gvs)
            self.train_op = self.optimizer.minimize(self.learning_loss)