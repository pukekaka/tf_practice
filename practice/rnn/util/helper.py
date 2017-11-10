class init_data:

    def __init__(self):
        self.mode = 'train'
        self.retore_training = False
        self.debut = False
        self.n_classes = 5
        self.seq_length = 50
        self.augment = True
        self.mode = 'MANN'
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
