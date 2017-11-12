import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os

class Model():
    def __init__(self,w_summary = True, logdir_train = None, logdir_test = None
                 ,batch_size = 16, drop_rate = 0.5, lear_rate = 2.5e-4, decay = 0.96,
                 decay_step = 2000, training = True,projection=True,r_number=20,idNumber=30):
        self.w_summary=w_summary
        self.logdir_train=logdir_train
        self.logdir_test=logdir_test
        self.batch_size=batch_size
        self.drop_rate=drop_rate
        self.learning_rate=lear_rate
        self.decay=decay
        self.decay_step=decay_step
        self.training=training
        self.projection=projection
        self.n_dict = {20:1, 32:2, 44:3, 56:4}
        self.r_number=r_number
        self.idNumber=idNumber
    def get_input(self):
        return self.img

    def get_output(self):
        return self.get_output

    def get_label(self):

        return self.label

    def get_loss(self):

        return self.loss

    def get_saver(self):


        return self.saver

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session()
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _init_session(self):
        """ Initialize Session
        """
        print('Session initialization')
        t_start = time.time()
        self.Session = tf.Session()

        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _center_loss_fn(self, embeddings, labels):
        embedding_size = embeddings.get_shape().as_list()[-1]
        centers = tf.get_variable(name='centers', shape=[self.hps.num_classes, embedding_size],
                                  initializer=tf.random_normal_initializer(stddev=0.1), trainable=False)
        label_indices = tf.argmax(labels, 1)
        centers_batch = tf.nn.embedding_lookup(centers, label_indices)
        center_loss = tf.nn.l2_loss(embeddings - centers_batch) / tf.to_float(tf.shape(embeddings)[0])
        new_centers = centers_batch - embeddings
        labels_unique, row_indices, counts = tf.unique_with_counts(label_indices)

        centers_update = tf.unsorted_segment_sum(new_centers, row_indices, tf.shape(labels_unique)[0]) / tf.to_float(
            counts)
        centers = tf.scatter_sub(centers, labels_unique, self.hps.ALPHA * centers_update)

        return center_loss

    def _weight_variable(self,shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def _softmax_layer(self,inpt, shape):
        fc_w = self._weight_variable(shape)
        fc_b = tf.Variable(tf.zeros([shape[1]]))

        fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)


        return fc_h

    def _conv_layer(self,inpt, filter_shape, stride):
        out_channels = filter_shape[3]

        filter_ = self._weight_variable(filter_shape)
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
        mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
        beta = tf.Variable(tf.zeros([out_channels]), name="beta")
        gamma = self._weight_variable([out_channels], name="gamma")

        batch_norm = tf.nn.batch_norm_with_global_normalization(
            conv, mean, var, beta, gamma, 0.001,
            scale_after_normalization=True)

        out = tf.nn.relu(batch_norm)

        return out

    def _residual_block(self,inpt, output_depth, down_sample, projection=False):
        input_depth = inpt.get_shape().as_list()[3]
        if down_sample:
            filter_ = [1, 2, 2, 1]
            inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

        conv1 = self._conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
        conv2 = self._conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

        if input_depth != output_depth:
            if projection:
                # Option B: Projection shortcut
                input_layer = self._conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
            else:
                # Option A: Zero-padding
                input_layer = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
        else:
            input_layer = inpt

        res = conv2 + input_layer

        return res


    def _graph_model(self,inputs):
        if self.r_number < 20 or (self.r_number - 20) % 12 != 0:
            print
            "ResNet depth invalid."
            return

        num_conv =(self.r_number - 20) / 12 + 1
        print(num_conv)
        num_conv=int(num_conv)
        print(num_conv,'==========num_conv')
        layers = []

        with tf.variable_scope('conv1'):
            conv1 = self._conv_layer(inputs, [3, 3, 3, 16], 1)
            layers.append(conv1)

        for i in range(num_conv):
            with tf.variable_scope('conv2_%d' % (i + 1)):
                conv2_x = self._residual_block(layers[-1], 16, False)
                conv2 = self._residual_block(conv2_x, 16, False)
                layers.append(conv2_x)
                layers.append(conv2)
            print(conv2.get_shape().as_list()[1:],'===conv2 list')
            #assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv3_%d' % (i + 1)):
                conv3_x = self._residual_block(layers[-1], 32, down_sample)
                conv3 = self._residual_block(conv3_x, 32, False)
                layers.append(conv3_x)
                layers.append(conv3)
            print(conv3.get_shape().as_list()[1:], '===conv3 list')

                #assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv4_%d' % (i + 1)):
                conv4_x = self._residual_block(layers[-1], 64, down_sample)
                conv4 = self._residual_block(conv4_x, 64, False)
                layers.append(conv4_x)
                layers.append(conv4)
            print(conv4.get_shape().as_list()[1:], '===conv4 list')

                # assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1, 2])
            assert global_pool.get_shape().as_list()[1:] == [64]
          #  self.center_loss=self._center_loss_fn()
            out = self._softmax_layer(global_pool, [64, self.idNumber])
            layers.append(out)

        print(layers[-1])
        return layers[-1]


    def generate_model(self):
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img=tf.placeholder(dtype= tf.float32, shape= (None, 256, 256, 3), name = 'input_img')
            self.label=tf.placeholder(dtype = tf.float32, shape = (None, self.idNumber))
        inputTime = time.time()
        print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
        self.output = self._graph_model(self.img)
        graphTime = time.time()
        print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
        with tf.name_scope('loss'):
            self.loss=-tf.reduce_sum(self.label*tf.log(self.output))
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
          #  self.acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accurTime = time.time()
        with tf.name_scope('steps'):
            self.train_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('lr'):
            self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                 staircase=True, name='learning_rate')
        lrTime = time.time()
        print('---LR : Done (' + str(int(abs(accurTime - lrTime))) + ' sec.)')

        with tf.name_scope('rmsprop'):
            self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        optimTime = time.time()
        print('---Optim : Done (' + str(int(abs(optimTime - lrTime))) + ' sec.)')
        with tf.name_scope('minimizer'):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
        minimTime = time.time()

        print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')
        self.init = tf.global_variables_initializer()
        initTime = time.time()

        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')
        with tf.name_scope('training'):
            tf.summary.scalar('loss', self.loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])

        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')

        self.weight_op = tf.summary.merge_all('weight')
        endTime = time.time()
        print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')

        del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, graphTime, inputTime

    def restore(self, load=None):
       
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary(summary=False)
                if load is not None:
                    print('Loading Trained Model')
                    t = time.time()
                    self.saver.restore(self.Session, load)
                    print('Model Loaded (', time.time() - t, ' sec.)')
                else:

                    print('Please give a Model in args (see README for further information)')






























