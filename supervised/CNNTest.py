import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 

warnings.filterwarnings("ignore")
supdir='../../data/supervised/'
predatadir='../../data/preprocess/'
source="Cs137" #"Co60"

shift=[-127,-210,-240,-320,-450,-530]
learning_rate=0
datasizex=10
datasizey=10
datachannel=14
outputdim=1
inputdim=1400
#input image placeholder
x=tf.placeholder("float",[None,datasizex,datasizey,datachannel])

conv1 = tf.layers.conv2d(x, 32, 2, activation=tf.nn.tanh)
# conv1 = tf.layers.max_pooling2d(conv1, 2, 1)

conv2 = tf.layers.conv2d(conv1, 64, 2, activation=tf.nn.tanh)
# # conv2 = tf.layers.max_pooling2d(conv2, 2, 1)
conv3 = tf.layers.conv2d(conv2, 64, 2, activation=tf.nn.relu)
conv4 = tf.layers.conv2d(conv2, 32, 2, activation=tf.nn.relu)
# # conv3 = tf.layers.max_pooling2d(conv2, 2, 2)

fc1 = tf.contrib.layers.flatten(conv3)

outputs = tf.contrib.layers.fully_connected(fc1,10,activation_fn=tf.nn.tanh )
outputs = tf.contrib.layers.dropout(fc1, keep_prob=0.9)

prediction = tf.contrib.layers.fully_connected(outputs,1)

saver = tf.train.Saver(max_to_keep=40)
init=tf.global_variables_initializer()


for noi in range(1):
    ModelDir=supdir+"CNNmodels"+str(noi+1)+'/'
    with open(predatadir+source+'normedwaveform'+str(noi+1)+'.dat', 'rb') as f:
        realwave=pickle.load(f)

    with tf.Session() as sess:
        sess.run(init)
        new_saver = tf.train.import_meta_graph(ModelDir+'test-10.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(ModelDir))

        testx=realwave['waveform'][:,800+shift[noi]:2200+shift[noi]]
        testpred=sess.run([prediction],feed_dict={y: inputy})

        with open(supdir+source+'predlabel'+str(noi+1)+'.dat', 'wb') as f:
            pickle.dump(testpred, f)
        

