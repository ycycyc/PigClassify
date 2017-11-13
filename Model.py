import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
import data_util
class Model():
    def __init__(self,w_summary = True, logdir_train = None, logdir_test = None
                 ,batch_size = 16, drop_rate = 0.5, lear_rate = 2.5e-4, decay = 0.96,
                 decay_step = 2000, training = True,projection=True,r_number=56,idNumber=30,ALPHA=0.5):
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
        self.ALPHA=ALPHA
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

    def get_center_loss(features, labels, alpha, num_classes):

        len_features = features.get_shape()[1]

        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])

        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)

        diff = centers_batch - features

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        return loss, centers, centers_update_op


    def _center_loss_fn(self, embeddings, labels):
        embedding_size = embeddings.get_shape().as_list()[-1]
        centers = tf.get_variable(name='centers', shape=[self.idNumber, embedding_size],
                                  initializer=tf.random_normal_initializer(stddev=0.1), trainable=False)
        label_indices = tf.argmax(labels, 1)
        centers_batch = tf.nn.embedding_lookup(centers, label_indices)
        center_loss = tf.nn.l2_loss(embeddings - centers_batch) / tf.to_float(tf.shape(embeddings)[0])
        new_centers = centers_batch - embeddings
        labels_unique, row_indices, counts = tf.unique_with_counts(label_indices)

        centers_update = tf.unsorted_segment_sum(new_centers, row_indices, tf.shape(labels_unique)[0]) / tf.to_float(
            counts)
        centers = tf.scatter_sub(centers, labels_unique, self.ALPHA * centers_update)

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
            conv1 = self._conv_layer(inputs, [3, 3, 3, 32], 1)
            layers.append(conv1)

        for i in range(num_conv):
            with tf.variable_scope('conv2_%d' % (i + 1)):
                conv2_x = self._residual_block(layers[-1], 32, False)
                conv2 = self._residual_block(conv2_x, 32, False)
                layers.append(conv2_x)
                layers.append(conv2)
            print(conv2.get_shape().as_list()[1:],'===conv2 list')
            #assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv3_%d' % (i + 1)):
                conv3_x = self._residual_block(layers[-1], 64, down_sample)
                conv3 = self._residual_block(conv3_x, 64, False)
                layers.append(conv3_x)
                layers.append(conv3)
            print(conv3.get_shape().as_list()[1:], '===conv3 list')

                #assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

        for i in range(num_conv):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv4_%d' % (i + 1)):
                conv4_x = self._residual_block(layers[-1], 128, down_sample)
                conv4 = self._residual_block(conv4_x, 128, False)
                layers.append(conv4_x)
                layers.append(conv4)
            print(conv4.get_shape().as_list()[1:], '===conv4 list')

                # assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1, 2])
          #  assert global_pool.get_shape().as_list()[1:] == [64]
          #  self.center_loss=self._center_loss_fn()
            fn=global_pool
            print(global_pool)
            out = self._softmax_layer(global_pool, [128, self.idNumber])
            layers.append(out)

        print(layers[-1])
        return layers[-1],fn


    def generate_model(self):
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img=tf.placeholder(dtype= tf.float32, shape= (None, 256, 256, 3), name = 'input_img')
            self.label=tf.placeholder(dtype = tf.float32, shape = (None, self.idNumber))
        inputTime = time.time()
        print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
        self.output,self.feature = self._graph_model(self.img)
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

    def _train(self, nEpochs=200, epochSize=5000, saveStep=500, validIter=10):
        data, label = data_util.readMyRecords("train.tfrecords")

        dataBatch, labelBatch = tf.train.shuffle_batch([data, label],
                                                       batch_size=self.batch_size, capacity=2000,
                                                       min_after_dequeue=1000)

        data2, label2 = data_util.readMyRecords("valid.tfrecords")
        dataBatch2, labelBatch2 = tf.train.shuffle_batch([data2, label2],
                                                         batch_size=self.batch_size, capacity=2000,
                                                         min_after_dequeue=1000)
        threads = tf.train.start_queue_runners(sess=self.Session)
        startTime = time.time()
        self.resume = {}
        self.resume['accur'] = []
        self.resume['loss'] = []
        self.resume['err'] = []

        for epoch in range(nEpochs):
            epochstartTime = time.time()
            avg_cost = 0.
            cost = 0.
            print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
            for i in range(epochSize):
                data_train, label_train = self.Session.run([dataBatch, labelBatch])
                percent = ((i + 1) / epochSize) * 100
                num = np.int(20 * percent / 100)
                tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                sys.stdout.write(
                    '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                  :4] + '%' + ' -cost: ' + str(
                        cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                sys.stdout.flush()
            if i % saveStep == 0:
                _, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op],
                                                 feed_dict={self.img: data_train, self.label: label_train})

                self.train_summary.add_summary(summary, epoch * epochSize + i)
                self.train_summary.flush()
            else:
                _, c, = self.Session.run([self.train_rmsprop, self.loss],
                                         feed_dict={self.img: data_train, self.label: label_train})
                cost += c
                avg_cost += c / epochSize
                epochfinishTime = time.time()
                # Save Weight (axis = epoch)

                weight_summary = self.Session.run(self.weight_op, {self.img: data_train, self.label: label_train})
                self.train_summary.add_summary(weight_summary, epoch)
                self.train_summary.flush()
                # self.weight_summary.add_summary(weight_summary, epoch)
                # self.weight_summary.flush()
                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                    int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                    ((epochfinishTime - epochstartTime) / epochSize))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session, os.path.join(os.getcwd(), str(self.name + '_' + str(epoch + 1))))
                self.resume['loss'].append(cost)
                # Validation Set
                accuracy_array = np.array([0.0] * 1)
                for i in range(validIter):
                    data_train2, label_train2 = self.Session.run([dataBatch2, labelBatch2])

                    # img_valid, gt_valid, w_valid = next(self.generator)
                    accuracy_pred = self.Session.run(self.acc,
                                                     feed_dict={self.img: data_train2, self.label: label_train2})
                    accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter
                print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%')
                self.resume['accur'].append(accuracy_pred)
                self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                valid_summary = self.Session.run(self.test_op,
                                                 feed_dict={self.img: data_train2, self.label: label_train2})
                self.test_summary.add_summary(valid_summary, epoch)
                self.test_summary.flush()
        print("Train Done")
        print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(
            nEpochs * epochSize * self.batchSize))
        print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(
            100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
        print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) + '%')
        print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))

    def training_init(self, nEpochs=200, epochSize=5000, saveStep=500, dataset=None, load=None):

        with tf.name_scope('Session'):
            self._init_weight()
#            self._define_saver_summary()
            if load is not None:
                self.saver.restore(self.Session, load)

            self._train(nEpochs, epochSize, saveStep, validIter=10)

























