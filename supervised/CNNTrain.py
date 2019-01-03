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
createmap=False
trainsize=30000
batchsize=6000 #5000
valsize=9290 #99000

accthre=0.5

stepNbofiter=5000
learningi=tf.placeholder("float",[1])
learning_rate=0.01*pow(0.1,learningi[0])

Nbofbatch=trainsize//batchsize
datasizex=10
datasizey=10
datachannel=14
outputdim=1
inputdim=1400
#input image placeholder
x=tf.placeholder("float",[None,datasizex,datasizey,datachannel])
y=tf.placeholder("float",[None,outputdim])

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

prediction = tf.contrib.layers.fully_connected(outputs,1) #,weights_regularizer= tf.contrib.layers.l2_regularizer(0.2))
loss = tf.losses.mean_squared_error(y,prediction)
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=40)
init=tf.global_variables_initializer()
fig = plt.figure(figsize=(9,7))

if createmap:
    Nboflearn=1
else:
    Nboflearn=4


for noi in range(1,6):
    ModelDir=supdir+"CNNmodels"+str(noi+1)+'/'
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
                inputx=artiwave['waveform'][startbatch*batchsize:(startbatch+1)*batchsize,800:2200].reshape((batchsize,datasizex,datasizey,datachannel))
                inputy=np.array(artiwave['para']['glabels'].iloc[startbatch*batchsize:(startbatch+1)*batchsize]).reshape(batchsize,1)+10
                sess.run(opt, feed_dict={x: inputx, y: inputy, learningi: thislearning})

                if iter % PrintCount == 0:
                    trainpred,trainloss=sess.run([prediction, loss],feed_dict={x: inputx, y: inputy, learningi: thislearning})
                    trainacc=sum((trainpred>(accthre+10))==(inputy-10).astype(np.int64))/len(trainpred)
                    listtrainacc.append(trainacc)
                    listtrainloss.append(trainloss)
                    valx=artiwave['waveform'][trainsize:trainsize+valsize,800:2200].reshape(valsize,datasizex,datasizey,datachannel)
                    valy=np.array(artiwave['para']['glabels'].iloc[trainsize:trainsize+valsize]).reshape(valsize,1)+10
                    valpred,valloss=sess.run([prediction, loss],feed_dict={x: valx, y: valy, learningi: thislearning})
                    valacc=sum((valpred>(accthre+10))==(valy-10).astype(np.int64))/len(valpred)
                    listvalacc.append(valacc)
                    listvalloss.append(valloss)

                    ax = plt.subplot2grid((2,2), (0,0))
                    ax.hist(trainpred-inputy,bins=np.linspace(-1,1,50),label='train',color='k',histtype='step')
                    ax.hist(valpred-valy,bins=np.linspace(-1.,1,50),label='validation',color='r',histtype='step')
                    plt.title('residual')
                    plt.legend(loc='upper right')
                    ax = plt.subplot2grid((2,2), (0,1))
                    ax.hist(inputy-10,bins=np.linspace(-1.5,3.5,30),label='truth',color='k',histtype='step')
                    ax.hist(trainpred-10,bins=np.linspace(-1.5,3.5,30),label='prediction',color='r',histtype='step')
                    plt.title('label')
                    plt.legend(loc='upper right')
                    ax = plt.subplot2grid((2,2), (1,0))
                    ax.plot(listtrainloss,label='train',color='k')
                    ax.plot(listvalloss,label='validation',color='r')
                    plt.title('loss')
                    plt.legend(loc='upper right')
                    ax = plt.subplot2grid((2,2), (1,1))
                    ax.plot(listtrainacc,label='train',color='k')
                    ax.plot(listvalacc,label='validation',color='r')
                    plt.title('Accurarcy')
                    plt.legend(loc='upper left')
                    if createmap==False:
                        plt.savefig(ModelDir+'plot'+str(learni)+'.pdf') #,dpi=200)
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
