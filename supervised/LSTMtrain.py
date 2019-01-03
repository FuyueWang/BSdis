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
source="Cs137" #"Co60"
createmap=True
trainsize=30000
batchsize=3000 #5000
valsize=9290 #99000

accthre=0.5

stepNbofiter=4000
Nbofpoint=1400
time_steps=140
n_input=10
n_classes=1

num_units=200


Nbofbatch=trainsize//batchsize

learningi=tf.placeholder("float",[1])
learning_rate=0.001*pow(0.1,learningi[0])

x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
#lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
lstm_layer=rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units,forget_bias=1), input_keep_prob=0.95, output_keep_prob=0.95, state_keep_prob=0.95)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
# print(len(outputs))
outputs = tf.contrib.layers.fully_connected(outputs[-1],16,weights_regularizer= tf.contrib.layers.l2_regularizer(0.1), activation_fn=tf.nn.tanh)

prediction = tf.contrib.layers.fully_connected(outputs,1 ,activation_fn=tf.nn.tanh)
loss = tf.losses.mean_squared_error(y,prediction)
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=40)
init=tf.global_variables_initializer()
fig = plt.figure(figsize=(9,5))

if createmap:
    Nboflearn=1
else:
    Nboflearn=6


for noi in range(1):
    ModelDir=supdir+"LSTMmodels"+str(noi+1)+'/'
    with open(supdir+source+'artificialwaveform'+str(noi+1)+'.dat', 'rb') as f:
        artiwave=pickle.load(f)
    for learni in range(0,Nboflearn):
        thislearning=[learni]
        if createmap:
            Nbofiter=21
            PrintCount=10
            ModelSaverCount=10
        else:
            Nbofiter=stepNbofiter*(1+learni*2)+10
            PrintCount=50
            ModelSaverCount=Nbofiter-10

        with tf.Session() as sess:
            sess.run(init)
            iter=0
            if createmap == False:
                new_saver = tf.train.import_meta_graph(ModelDir+'test-10.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(ModelDir))
            listtrainloss=[]
            listvalloss=[]
            listtrainacc=[]
            listvalacc=[]            
            while iter<Nbofiter:
                startbatch=iter%Nbofbatch
                inputx=artiwave['waveform'][startbatch*batchsize:(startbatch+1)*batchsize,800:2200].reshape((batchsize,time_steps,n_input))
                inputy=np.array(artiwave['para']['glabels'].iloc[startbatch*batchsize:(startbatch+1)*batchsize]).reshape(batchsize,1)+10
                sess.run(opt, feed_dict={x: inputx, y: inputy, learningi: thislearning})

                if iter % PrintCount == 0:
                    trainpred,trainloss=sess.run([prediction, loss],feed_dict={x: inputx, y: inputy, learningi: thislearning})
                    trainacc=1-sum((trainpred>accthre)==(inputy-10).astype(np.int64))/len(trainpred)
                    listtrainacc.append(trainacc)
                    listtrainloss.append(trainloss)
                    valx=artiwave['waveform'][trainsize:trainsize+valsize,800:2200].reshape(valsize,time_steps,n_input)
                    valy=np.array(artiwave['para']['glabels'].iloc[trainsize:trainsize+valsize]).reshape(valsize,1)+10
                    valpred,valloss=sess.run([prediction, loss],feed_dict={x: valx, y: valy, learningi: thislearning})
                    valacc=1-sum((valpred>accthre)==(valy-10).astype(np.int64))/len(valpred)
                    listvalacc.append(valacc)
                    listvalloss.append(valloss)

                    ax = plt.subplot2grid((1,2), (0,0))
                    ax.plot(listtrainloss,label='train',color='k')
                    ax.plot(listvalloss,label='validation',color='r')
                    plt.title('loss')
                    plt.legend(loc='upper right')
                    ax = plt.subplot2grid((1,2), (0,1))
                    ax.plot(listtrainacc,label='train',color='k')
                    ax.plot(listvalacc,label='validation',color='r')
                    plt.title('Accurarcy')
                    plt.legend(loc='upper right')
                    plt.draw()
                    plt.pause(0.001)
                    
                    print("Noise Collection",noi,"learning rate",sess.run(learning_rate,feed_dict={x:inputx, y: inputy, learningi: thislearning}))
                    print("For iter",iter,"train: Loss",trainloss ,"acc",trainacc[0])
                    print("For iter",iter,"val: Loss",valloss,"acc",valacc[0])

                if iter % ModelSaverCount ==0:
                    if createmap:
                        saver.save(sess, ModelDir+"test",global_step=iter,write_meta_graph=True)
                    else:
                        saver.save(sess, ModelDir+"test",global_step=iter,write_meta_graph=False)
                iter=iter+1
