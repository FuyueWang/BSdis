import tensorflow as tf
# class mynetwork:
#     @abstractmethod
#     def __init__(self,):


class mycnnnetwork():
    """
    define every specific network
    """
    def __init__(self,dataname,modeldir,flag):
        self.trainsize=30000
        self.batchsize=6000 
        self.validatesize=9290 
        
        self.datadim=[10,10,14,1] #[imput dim0 dim1..., output dim]

        self.startlearningrate= 0.0005
        self.classificationthreshold = 0.5 # set to be 999 for regression
        self.Nbofsaveriteractions = 4000
        self.Nboflearn=3

        if flag<4:
            self.cnnlayer=[32,80,64]
            self.cnnact=[tf.nn.tanh,tf.nn.tanh,tf.nn.relu]
            self.dnnlayer=[self.datadim[-1]]
            self.dnnact=[tf.nn.tanh]
        elif flag==4:
            self.cnnlayer=[32,32,32]
            self.cnnact=[tf.nn.tanh,tf.nn.tanh,tf.nn.relu]
            self.dnnlayer=[20,self.datadim[-1]]
            self.dnnact=[tf.nn.tanh,tf.nn.relu]
        elif flag==5:
            self.cnnlayer=[32,16,8]
            self.cnnact=[tf.nn.tanh,tf.nn.tanh,tf.nn.relu]
            self.dnnlayer=[self.datadim[-1]]
            self.dnnact=[tf.nn.tanh]
        elif flag==6:
            self.cnnlayer=[16,8,8]
            self.cnnact=[tf.nn.tanh,tf.nn.tanh,tf.nn.relu]
            self.dnnlayer=[3,self.datadim[-1]]
            self.dnnact=[tf.nn.tanh,tf.nn.relu]
    
        self.lstmlayer=[]


class mylstmnetwork():
    """
    define every specific network
    """
    def __init__(self,dataname,modeldir,flag):
        self.trainsize=30000
        self.batchsize=6000 
        self.validatesize=9290 
        
        self.datadim=[70,20,1] #[imput dim0 dim1..., output dim] [time_steps,n_input]

        self.startlearningrate= 0.001
        self.classificationthreshold = 0.5 # set to be 999 for regression
        self.Nbofsaveriteractions = 4000
        self.Nboflearn=3

        if flag<4:
            self.lstmlayer=[150]
            self.dnnlayer=[10,self.datadim[-1]]
            self.dnnact=[tf.nn.tanh,tf.nn.relu]
        elif flag==4:
            self.lstmlayer=[200]
            self.dnnlayer=[20,self.datadim[-1]]
            self.dnnact=[tf.nn.tanh,tf.nn.relu]           
        elif flag==5:
            self.lstmlayer=[200]
            self.dnnlayer=[20,self.datadim[-1]]
            self.dnnact=[tf.nn.tanh,tf.nn.relu]           
        elif flag==6:
            self.lstmlayer=[200]
            self.dnnlayer=[20,self.datadim[-1]]
            self.dnnact=[tf.nn.tanh,tf.nn.relu]            
    
        self.cnnlayer=[]
        self.cnnact=[]

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
