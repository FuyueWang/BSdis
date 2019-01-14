import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path

import dataclass
import architecture
import plotfunction

class trainmodel():
    """
    training related 
    """
    def __init__(self, trainpara, data, netarchitecture):
        self.startlearningrate = trainpara[0]
        self.Nbofsaveriteractions = trainpara[1]
        self.modeldir = trainpara[2]
        self.Nboflearn = trainpara[3]
        self.accthre = trainpara[4]
        self.traindata = data
        self.netarchitecture = netarchitecture
        
    def train(self):
        x = tf.placeholder("float",[None]+self.traindata.datadim[:-1])
        y = tf.placeholder("float",[None]+[self.traindata.datadim[-1]])

        learningi=tf.placeholder("float",[1])
        learning_rate=self.startlearningrate*pow(0.1,learningi[0])

        prediction = self.netarchitecture.network(x)
        loss = tf.losses.mean_squared_error(y,prediction)
        opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        init=tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=4)
        
        fig = plt.figure(figsize=(9,7))

        for learni in range(self.Nboflearn):
            thislearning=[learni]

            Nbofiter=self.Nbofsaveriteractions*(1+learni*2)+10
            PrintCount=50
            ModelSaverCount=self.Nbofsaveriteractions-10

            with tf.Session() as sess:
                sess.run(init)
                iter=0

                if os.path.exists(self.modeldir+'test-0.meta'):
                    new_saver = tf.train.import_meta_graph(self.modeldir+'test-0.meta')
                    new_saver.restore(sess, tf.train.latest_checkpoint(self.modeldir))
                else:
                    saver.save(sess, self.modeldir+"test",global_step=iter,write_meta_graph=True)
       
                listtrainloss=[]
                listvalloss=[]
                listtrainresult=[]
                listvalresult=[]

                iter=0
                while iter<Nbofiter:
                    startbatch=iter%self.traindata.Nbofbatch
                    inputx,inputy = self.traindata.Gettraindata(startbatch)
                    sess.run(opt, feed_dict={x: inputx, y: inputy, learningi: thislearning})
                    
                    if iter % PrintCount == 0:
                        trainpred,trainloss,lrate=sess.run([prediction, loss,learning_rate],feed_dict={x: inputx, y: inputy, learningi: thislearning})
                        valx,valy=self.traindata.Getvaldata()
                        valpred,valloss=sess.run([prediction, loss],feed_dict={x: valx, y: valy, learningi: thislearning})
                        if self.accthre < 998:
                            trainresult=sum((trainpred>(self.accthre))==(inputy).astype(np.int64))/len(trainpred)
                            valresult=sum((valpred>(self.accthre))==(valy).astype(np.int64))/len(valpred)
                        listtrainresult.append(trainresult)
                        listtrainloss.append(trainloss)
                        listvalresult.append(valresult)
                        listvalloss.append(valloss)

                        plotfunction.plotclasstrain(self.GenerateROCdata(valy,valpred),[inputy,trainpred],[listtrainloss,listvalloss],[listtrainresult,listvalresult],self.modeldir+'plot'+str(learni)+'.pdf')
                            
                        print("learning rate",lrate)
                        print("For iter",iter,"train: Loss",trainloss ,"res",trainresult[0])
                        print("For iter",iter,"val: Loss",valloss,"res",valresult[0])
                    if iter % ModelSaverCount ==0:
                        saver.save(sess, self.modeldir+"test",global_step=iter,write_meta_graph=False)
                    iter=iter+1
                    

    def GenerateROCdata(self,truthlabel,predictedlabel):
        rocdf=pd.DataFrame()
        rocdf['truth']=list(truthlabel[:,0].astype(np.int32))  
        rocdf['pred']=list(predictedlabel[:,0])
        tp=[]
        fp=[]
        for k in range(100):
            threshold=0.01*k
            tp.append(sum(rocdf[rocdf['truth']==1]['pred']>threshold)/rocdf[rocdf['truth']==1].shape[0])
            fp.append(sum(rocdf[rocdf['truth']==0]['pred']>threshold)/rocdf[rocdf['truth']==1].shape[0])
        return [fp,tp]
