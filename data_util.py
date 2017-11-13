import  tensorflow as tf
import cv2
import os
import sys
import os.path
import numpy as np
rootdir = "./image/"

def makeData():
    train_writer=tf.python_io.TFRecordWriter('train.tfrecords')
    valid_writer=tf.python_io.TFRecordWriter('valid.tfrecords')

    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                label=np.zeros(shape=[30],dtype=np.uint8)
                print(filename)
                img=cv2.imread(os.path.join(parent,filename))
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img=cv2.resize(img,(256,256))
               # img=img.astype
                result = filename.split('_')[0]
                rn=int(filename.split('_')[1][1:-4])
                print('rn:  ',rn)
                label[int(result)-1]=1
                print(label)
                #print(img.shape)
             #   img_raw=img.astype(np.uint8)
                img_raw=img.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
                    'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))

                if rn%5==0:
                    valid_writer.write(example.SerializeToString())
                else:
                    train_writer.write(example.SerializeToString())

                #cv2.imshow('w',img)
            #cv2.waitKey(100)


#makeData()

def readMyRecords(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string),
                                       })

    data = tf.decode_raw(features['data_raw'], tf.uint8)
    data = tf.reshape(data, [256, 256, 3])
    #data = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.decode_raw(features['label'], tf.uint8)

    label =tf.reshape(label,[30])
  #  label=tf.reshape(label,[1])

    return data,label

def sdf():
    data, label = readMyRecords("train.tfrecords")

    data_batch, label_batch = tf.train.shuffle_batch([data, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l = sess.run([data_batch, label_batch])
            # l = to_categorical(l, 12)
            print(val.shape, l)

