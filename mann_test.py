
import tensorflow as tf

from util.model import MANN
from util.helper import *
from tensorflow.python import debug as tf_debug


class InitValue:
    def __init__(self):
        self.mode = 'train'
        self.restore_training = False
        self.debug = False
        self.label_type = 'one_hot'
        self.n_classes = 5
        self.seq_length = 50
        self.augment = True
        self.model = 'MANN'
        self.read_head_num = 4
        self.batch_size = 16
        self.num_epoches = 100000
        self.learning_rate = 1e-3
        self.rnn_size = 200
        self.image_width = 20
        self.image_height = 20
        self.rnn_num_layers = 1
        self.memory_size = 128
        self.memory_vector_dim = 40
        self.test_batch_num = 100
        self.n_train_classes = 1200
        self.n_test_classes = 423
        self.save_dir = './save/one_shot_learning'
        self.tensorboard_dir = './summary/one_shot_learning'


def test_f(args, y, output):
    correct = [0] * args.seq_length
    total = [0] * args.seq_length
    if args.label_type == 'one_hot':
        y_decode = one_hot_decode(y)
        output_decode = one_hot_decode(output)
    elif args.label_type == 'five_hot':
        y_decode = five_hot_decode(y)
        output_decode = five_hot_decode(output)
    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]


def test(iv):
    model = MANN(iv)
    data_loader = OmniglotDataLoader(
        image_size=(iv.image_width, iv.image_height),
        n_train_classses=iv.n_train_classes,
        n_test_classes=iv.n_test_classes
    )
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(iv.save_dir + '/' + iv.model)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
        y_list = []
        output_list = []
        loss_list = []
        for b in range(iv.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length,
                                                          type='test',
                                                          augment=iv.augment,
                                                          label_type=iv.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
            y_list.append(y)
            output_list.append(output)
            loss_list.append(learning_loss)
        accuracy = test_f(iv, np.concatenate(y_list, axis=0), np.concatenate(output_list, axis=0))
        for accu in accuracy:
            print('%.4f' % accu, end='\t')
        print(np.mean(loss_list))


def learn():
    iv = InitValue()
    # print (iv.batch_size)
    if iv.mode == 'train':
        model = MANN(iv)

        data_loader = OmniglotDataLoader(
            image_size=(iv.image_width, iv.image_height),
            n_train_classses=iv.n_train_classes,
            n_test_classes=iv.n_test_classes
        )
        with tf.Session() as sess:
            if iv.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if iv.restore_training:
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(iv.save_dir + '/' + iv.model)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver = tf.train.Saver(tf.global_variables())
                tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter(iv.tensorboard_dir + '/' + iv.model, sess.graph)
            print(iv)
            print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")
            for b in range(iv.num_epoches):

                # Test

                if b % 100 == 0:
                    x_image, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length,
                                                                  type='test',
                                                                  augment=iv.augment,
                                                                  label_type=iv.label_type)
                    feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
                    output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                    merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                    train_writer.add_summary(merged_summary, b)
                    # state_list = sess.run(model.state_list, feed_dict=feed_dict)  # For debugging
                    # with open('state_long.txt', 'w') as f:
                    #     print(state_list, file=f)
                    accuracy = test_f(iv, y, output)
                    for accu in accuracy:
                        print('%.4f' % accu, end='\t')
                    print('%d\t%.4f' % (b, learning_loss))

                # Save model

                if b % 5000 == 0 and b > 0:
                    saver.save(sess, iv.save_dir + '/' + iv.model + '/model.tfmodel', global_step=b)

                # Train

                x_image, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length,
                                                              type='train',
                                                              augment=iv.augment,
                                                              label_type=iv.label_type)
                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
                sess.run(model.train_op, feed_dict=feed_dict)


if __name__ == '__main__':
    # print('test')
    learn()