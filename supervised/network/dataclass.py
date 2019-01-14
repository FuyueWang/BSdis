import numpy as np
import pickle
class dataclass():
    """
    operactions with data
    """
    def __init__(self, dataname, dimpara, datadim):
        self.trainsize = dimpara[0]
        self.batchsize = dimpara[1]
        self.validatesize = dimpara[2]
        self.Nbofbatch=self.trainsize//self.validatesize
        self.datadim = datadim #[imput dim0 dim1..., output dim]
        with open(dataname[0], 'rb') as f:
            self.data=pickle.load(f)
        self.featurename = dataname[1]
        self.labelitem = dataname[2]
        self.labelname = dataname[3]
        print(self.data[self.featurename].shape)
        
    def Gettraindata(self,startbatch=0):
        inputx = self.data[self.featurename][startbatch*self.batchsize:(startbatch+1)*self.batchsize,800:2200].reshape([self.batchsize]+self.datadim[:-1])
        inputy = np.array(self.data[self.labelitem][self.labelname].iloc[startbatch*self.batchsize:(startbatch+1)*self.batchsize]).reshape(self.batchsize,1)
        return inputx, inputy

    def Getvaldata(self):
        valx = self.data[self.featurename][self.trainsize:self.trainsize+self.validatesize,800:2200].reshape([self.validatesize]+self.datadim[:-1])
        valy = np.array(self.data[self.labelitem][self.labelname].iloc[self.trainsize:self.trainsize+self.validatesize]).reshape(self.validatesize,1)
        return valx, valy
