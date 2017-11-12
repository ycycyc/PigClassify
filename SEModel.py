import time
import tensorflow as tf
import sys
import datetime
import os
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
class SEModel():
    def __init__(self, w_summary=True, logdir_train=None, logdir_test=None
                 , batch_size=16, drop_rate=0.5, lear_rate=2.5e-4, decay=0.96,
                 decay_step=2000, training=True, projection=True, r_number=56, idNumber=30, ALPHA=0.5,momentum=0.9
                 ,cardinality=8,ratio=1,ModelName='SeModel01'):
        self.name=ModelName
        self.w_summary = w_summary
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.decay_step = decay_step
        self.training = training
        self.projection = projection
        self.r_number = r_number
        self.ratio=ratio
        self.idNumber = idNumber
        self.CENTER_LOSS_ALPHA= ALPHA
        self.momentum=momentum
        self.cardinality = cardinality  # how many split ?
        self.blocks = 3  # res_block ! (split + transition)
        self.depth = 64  # out channel
        self.reduction_ratio = 4
        self.iteration = 391
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

    def _conv_layer(self,input, filter, kernel, stride, padding='SAME', layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                       padding=padding)

        return network

    def _Global_Average_Pooling(self,x):
        return global_avg_pool(x, name='Global_avg_pooling')

    def _Average_pooling(self,x, pool_size=[2, 2], stride=2, padding='SAME'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


    def _Relu(self,x):
        return tf.nn.relu(x)

    def _Sigmoid(self,x):
        return tf.nn.sigmoid(x)

    def _Concatenation(self,layers):
        return tf.concat(layers, axis=3)

    def _Fully_connected(self,x, uints,layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=False, units=uints)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = self._conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope + '_conv1')
            x=tf.layers.batch_normalization(x,training=self.training,name=scope+'_batch1')
            #x = self._Batch_Normalization(x,scope=scope + '_batch1')
            x = self._Relu(x)

        return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = self._conv_layer(x, filter=self.depth, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
            x=tf.layers.batch_normalization(x,training=self.training,name=scope+'_batch1')
            x = self._Relu(x)
            x = self._conv_layer(x, filter=self.depth, kernel=[3, 3], stride=stride, layer_name=scope + '_conv2')
            x=tf.layers.batch_normalization(x,training=self.training,name=scope+'_batch2')
            x = self._Relu(x)
        return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = self._conv_layer(x, filter=out_dim, kernel=[1, 1], stride=1, layer_name=scope + '_conv1')
            x=tf.layers.batch_normalization(x,training=self.training,name=scope+'_batch1')
                # x = Relu(x)

        return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

        return self._Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = self._Global_Average_Pooling(input_x)
            excitation=tf.layers.dense(squeeze,units=out_dim / ratio,name=layer_name + '_fully_connected1')
           # excitation = self._Fully_connected(squeeze, units=out_dim / ratio,
                                           #  layer_name=layer_name + '_fully_connected1')
            excitation = self._Relu(excitation)
            excitation=tf.layers.dense(excitation, units=out_dim, name=layer_name + '_fully_connected2')
           # excitation = self._Fully_connected()
            excitation = self._Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

        return scale

    def residual_layer(self, input_x, out_dim, layer_num):
            # split + transform(bottleneck) + transition + merge
            # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(self.blocks):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=self.reduction_ratio,
                                                  layer_name='squeeze_layer_' + layer_num + '_' + str(i))

            if flag is True:
                pad_input_x =self._Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x,
                                         [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
            else:
                pad_input_x = input_x

            input_x = self._Relu(x + pad_input_x)

        return x

    def get_center_loss(self,features, labels, alpha, num_classes):

        len_features = features.get_shape()[1]
        print(len_features,'-len feature')
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        print(labels,'   labels')

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


    def _graph_model(self, inputs):
        input_x = self.first_layer(inputs, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        x = self.residual_layer(x, out_dim=256, layer_num='3')
        x = self.residual_layer(x, out_dim=512, layer_num='4')

        x = self._Global_Average_Pooling(x)

        x = flatten(x)
        x=tf.layers.dropout(x,rate=self.drop_rate,training=self.training)
        feature=tf.layers.dense(x,units=512,name='feature_fn_layer')
        x=tf.layers.dense(x,name='final_fully_connected',units=self.idNumber)
        #x = self._Fully_connected(x, layer_name='final_fully_connected')

        return x,feature


    def generate_model(self):
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')
            self.label = tf.placeholder(dtype=tf.int64, shape=(None, self.idNumber))
        inputTime = time.time()
        print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
        self.output,self.feature = self._graph_model(self.img)
        graphTime = time.time()
        print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
        with tf.name_scope('loss'):
            with tf.name_scope('center_loss'):
                with tf.name_scope('center_loss'):
                    self.center_loss, self.centers, self.centers_update_op =self.get_center_loss(self.feature, self.label, self.CENTER_LOSS_ALPHA,
                                                                                  self.idNumber)
                with tf.name_scope('softmax_loss'):
                    self.softmax_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.output))
                with tf.name_scope('total_loss'):
                    self.total_loss = self.softmax_loss + self.ratio * self.center_loss

                # center_loss, centers, centers_update_op = self.get_center_loss(self.feature, self.label, self.ALPHA)
                #   with tf.name_scope('softmax_loss'):
                #  softmax_loss = tf.reduce_mean(
                #      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self, logits=logits))
                #  with tf.name_scope('total_loss'):
                #
                # total_loss = softmax_loss + ratio * center_loss
        with tf.name_scope('accuracy'):
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.output, 1), self.label), tf.float32))
            #  self.acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        with tf.name_scope('loss/'):
                tf.summary.scalar('CenterLoss', self.center_loss)
                tf.summary.scalar('SoftmaxLoss', self.softmax_loss)
                tf.summary.scalar('TotalLoss',self.total_loss)
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
            with tf.control_dependencies([self.centers_update_op]):
                self.train_rmsprop = self.rmsprop.minimize(self.total_loss, self.train_step)
        minimTime = time.time()

        print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')
        self.init = tf.global_variables_initializer()
        initTime = time.time()

        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')
        with tf.name_scope('training'):
            tf.summary.scalar('CenterLoss', self.total_loss, collections=['train'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])
           # tf.summary.scalar('CenterLoss', center_loss)
            tf.summary.scalar('SoftmaxLoss',self.softmax_loss)

            tf.summary.scalar('TotalLoss', self.total_loss)
        with tf.name_scope('summary'):
            tf.summary.scalar('Acc', self.acc, collections=['train', 'test'])
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

    def _train(self, nEpochs = 10, epochSize = 1000, saveStep = 500, validIter = 10):
        with tf.name_scope('Train'):

            self.generator = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'train')
            self.valid_gen = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'valid')
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
                # Training Set
                for i in range(epochSize):
                    percent = ((i+1)/epochSize) * 100
                    num = np.int(20*percent/100)
                    tToEpoch = int((time.time() - epochstartTime) * (100 - percent)/(percent))
                    sys.stdout.write('\r Train: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                    sys.stdout.flush()
                    img_train, gt_train, weight_train = next(self.generator)
                    if i % saveStep == 0:

                        _, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.img : img_train, self.label: gt_train})

                        self.train_summary.add_summary(summary, epoch*epochSize + i)
                        self.train_summary.flush()
                    else:

                        _, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.img : img_train, self.label: gt_train})
                    cost += c
                    avg_cost += c/epochSize
                epochfinishTime = time.time()
                #Save Weight (axis = epoch)

                weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.label: gt_train})
                self.train_summary.add_summary(weight_summary, epoch)
                self.train_summary.flush()
                #self.weight_summary.add_summary(weight_summary, epoch)
                #self.weight_summary.flush()
                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime-epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime-epochstartTime)/epochSize))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session, os.path.join(os.getcwd(),str(self.name + '_' + str(epoch + 1))))
                self.resume['loss'].append(cost)
                # Validation Set
                accuracy_array = np.array([0.0]*1)
                for i in range(validIter):
                    img_valid, gt_valid, w_valid = next(self.generator)
                    accuracy_pred = self.Session.run(self.acc, feed_dict = {self.img : img_valid, self.label: gt_valid})
                    accuracy_array += np.array(accuracy_pred, dtype = np.float32) / validIter
                print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
                self.resume['accur'].append(accuracy_pred)
                self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                valid_summary = self.Session.run(self.test_op, feed_dict={self.img : img_valid, self.label: gt_valid})
                self.test_summary.add_summary(valid_summary, epoch)
                self.test_summary.flush()
        print("Train Done")
        print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize) )
        print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
        print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
        print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - startTime)))

    def record_training(self, record):

        out_file = open(self.name + '_train_record.csv', 'w')
        for line in range(len(record['accur'])):
            out_string = ''
            labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
            for label in labels:
                out_string += str(label) + ', '
            out_string += '\n'
            out_file.write(out_string)
        out_file.close()
        print('Training Record Saved')

    def training_init(self, nEpochs=10, epochSize=1000, saveStep=500, dataset=None, load=None):

        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()
                if load is not None:
                    self.saver.restore(self.Session, load)

                self._train(nEpochs, epochSize, saveStep, validIter=10)






















