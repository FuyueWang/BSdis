import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 

import architecture 
import dataclass
import trainmodel
import mynetwork

supdir='../../../data/supervised/'
source="Cs137" #"Co60"

def main():
    noi = 1
    # specific parameters
    dataname=[supdir+source+'artificialwaveform'+str(noi)+'.dat', 'waveform', 'para', 'glabels']

    # CNN
    # modeldir=supdir+"CNNmodels"+str(noi)+'/'
    # mynet=mynetwork.mycnnnetwork(dataname,modeldir,noi)

    # lstm
    modeldir=supdir+"LSTMmodels"+str(noi)+'/'
    mynet=mynetwork.mylstmnetwork(dataname,modeldir,noi)

    
    #  ----------------------------------------------------------------------#
    # in netowrk sessions. DO NOT CHANGE !!!!!!!!!!!!
    netarchitecture = architecture.architecture(dict(layer=mynet.cnnlayer,activation=mynet.cnnact),dict(layer=mynet.lstmlayer),dict(layer=mynet.dnnlayer,activation=mynet.dnnact))
    data = dataclass.dataclass(dataname,[mynet.trainsize,mynet.batchsize,mynet.validatesize],mynet.datadim)
    training = trainmodel.trainmodel([mynet.startlearningrate, mynet.Nbofsaveriteractions, modeldir, mynet.Nboflearn, mynet.classificationthreshold],data, netarchitecture)
    training.train()
    #  ----------------------------------------------------------------------#

if __name__ == '__main__':
    # args = argsparser()
    main()
