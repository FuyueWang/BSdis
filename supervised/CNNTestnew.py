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



def cnnnetwork(x,layer=[32,64,32,32]):
    conv = tf.layers.conv2d(x, layer[0], 2, activation=tf.nn.tanh)
    for k in range(1,len(layer)):
        if k<len(layer)-1:
            conv = tf.layers.conv2d(conv, layer[k], 2, activation=tf.nn.tanh)
        else:
            conv = tf.layers.conv2d(conv, layer[k], 2, activation=tf.nn.relu)
        # if k ==1:
        #     conv = tf.layers.max_pooling2d(conv, 2, 2)
    conv = tf.contrib.layers.flatten(conv)
    return conv

def dnnnetwork(x,layer=[10,1]):
    fc = tf.contrib.layers.dropout(x, keep_prob=0.9)
    for k in range(len(layer)):
        fc = tf.contrib.layers.fully_connected(fc,layer[k],activation_fn=tf.nn.tanh ) #relu
        # fc = tf.contrib.layers.dropout(fc, keep_prob=0.9)
    return fc




shift=[0,-127,-210,-240,-320,-450,-530]
learning_rate=0
datasizex=10
datasizey=10
datachannel=14
outputdim=1
inputdim=1400


for noi in range(5,6):
    #input image placeholder
    x=tf.placeholder("float",[None,datasizex,datasizey,datachannel])

    if noi<4:
        cnnlayer=[32,80,64]
        dnnlayer=[outputdim]
    elif noi==4:
        cnnlayer=[32,32,32]
        dnnlayer=[20,outputdim]
    elif noi==5:
        cnnlayer=[32,16,8]
        dnnlayer=[outputdim]
    elif noi==6: #output dnn layer is relu, others are tanh
        cnnlayer=[16,8,8]
        dnnlayer=[3,outputdim]
        
    prediction=dnnnetwork(cnnnetwork(x,cnnlayer),dnnlayer)

    saver = tf.train.Saver(max_to_keep=40)
    init=tf.global_variables_initializer()

    ModelDir=supdir+"CNNmodels"+str(noi)+'/'
    with open(predatadir+source+'normedwaveform'+str(noi)+'.dat', 'rb') as f:
        realwave=pickle.load(f)

    with tf.Session() as sess:
        sess.run(init)
        new_saver = tf.train.import_meta_graph(ModelDir+'test-10.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(ModelDir))

        testx=realwave['waveform'][:,800+shift[noi]:2200+shift[noi]].reshape((realwave['waveform'].shape[0],datasizex,datasizey,datachannel))
        testpred=sess.run([prediction],feed_dict={x: testx})[0]

        paradf=realwave['para']
        paradf['glabels']=testpred
        with open(supdir+source+'predlabel'+str(noi)+'.dat', 'wb') as f:
            pickle.dump(paradf, f)
        

