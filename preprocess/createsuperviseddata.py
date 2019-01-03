import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from array import array
import pandas as pd
import os.path
textsize=15
matplotlib.rcParams.update({'font.size': textsize})
predatadir='../../data/preprocess/'
unsupdir='../../data/unsupervised/'
supdir='../../data/supervised/'
plotdir='../../plot/preprocess/'
source="Cs137" #"Co60"
noise=[0,0.025,0.05,0.07,0.1,0.2,0.4]



def createartificialdataforsupervised():
    Nbofplotnoisepoint=3000
    with open(unsupdir+source+'testresultlabel.dat', 'rb') as f:
        para = pickle.load(f)
    with open(predatadir+source+'normedwaveform0.dat', 'rb') as f:
        wave = pickle.load(f)
    paracolumnname=['idev','ampli','ped','cross','slope','tau','noise','glabels']
    for noi in range(6):
        with open(predatadir+source+'noisecollection'+str(noi+1)+'.dat', 'rb') as f:
            noisecollection = pickle.load(f)
        noisewave=[]
        startid=list(map(int,np.round(np.random.uniform(0,len(noisecollection)-Nbofplotnoisepoint,wave['waveform'].shape[0]))))
        for k in range(len(startid)):
            noisewave+=list(noisecollection[startid[k]:startid[k]+3000])
            if k%10000==0:
                print(noi,k)
        waveform=wave['waveform']+np.array(noisewave).reshape(len(startid),3000)

        waveform=np.concatenate((waveform,para),axis=1)
        np.random.shuffle(waveform)
        paradf=pd.DataFrame()
        for i in range(8):
            paradf[paracolumnname[i]]=waveform[:,-8+i]
        dictdata=dict(waveform=waveform[:,:-8],para=paradf)
        with open(supdir+source+'artificialwaveform'+str(noi+1)+'.dat', 'wb') as f:
            pickle.dump(dictdata, f)


# createartificialdataforsupervised()

def plotartificialonly():
    fig = plt.figure(figsize=(10,7))
    artilist=[]
    for noi in range(6):
        with open(supdir+source+'artificialwaveform'+str(noi+1)+'.dat', 'rb') as f:
            artiwave=pickle.load(f)
        print(noi,artiwave['waveform'].shape)
        artilist.append(artiwave)

    eveid=int(np.round(np.random.uniform(0,artilist[0]['waveform'].shape[0])))
    for noi in range(6):
        ax = plt.subplot2grid((2,3), (noi//3,noi%3))
        ax.plot(artilist[noi]['waveform'][eveid,:],label='gmm: '+str(artilist[noi]['para']['glabels'].iloc[eveid]))
        if noi <5:
            plt.title(str(noise[noi+1])+'<noise<'+str(noise[noi+2]),fontsize=textsize)
        else:
            plt.title('noise>'+str(noise[noi+1]),fontsize=textsize)
        plt.xlabel('time')
        plt.ylabel('wave')
        plt.ylim([-3,4])
        plt.legend(loc='upper right') #,fontsize=plottextsize)
    plt.subplots_adjust(left=0.07,bottom=0.08,right=0.98,top=0.95,wspace=0.3,hspace=0.35)
    plt.savefig(plotdir+'artificial.pdf')
    plt.show()


def plotrealandartificial():
    fig = plt.figure(figsize=(10,7))
    artilist=[]
    reallist=[]
    shift=[-127,-210,-240,-320,-450,-530]
    for noi in range(6):
        with open(supdir+source+'artificialwaveform'+str(noi+1)+'.dat', 'rb') as f:
            artiwave=pickle.load(f)
        print(noi,artiwave['waveform'].shape)
        artilist.append(artiwave)
        with open(predatadir+source+'normedwaveform'+str(noi+1)+'.dat', 'rb') as f:
            realwave=pickle.load(f)
        reallist.append(realwave)
    for noi in range(6):
        ax = plt.subplot2grid((2,3), (noi//3,noi%3))
        for k in range(16):
            if k==0:
                eveid=int(np.round(np.random.uniform(0,reallist[noi]['waveform'].shape[0])))
                ax.plot(reallist[noi]['waveform'][eveid,800+shift[noi]:2200+shift[noi]],color='k',label='truth')
                eveid=int(np.round(np.random.uniform(0,artilist[noi]['waveform'].shape[0])))
                ax.plot(artilist[noi]['waveform'][eveid,800:2200],color='r',label='artificial')
            else:
                eveid=int(np.round(np.random.uniform(0,reallist[noi]['waveform'].shape[0])))
                ax.plot(reallist[noi]['waveform'][eveid,800+shift[noi]:2200+shift[noi]],color='k')
                eveid=int(np.round(np.random.uniform(0,artilist[noi]['waveform'].shape[0])))
                ax.plot(artilist[noi]['waveform'][eveid,800:2200],color='r')
        if noi <5:
            plt.title(str(noise[noi+1])+'<noise<'+str(noise[noi+2]),fontsize=textsize)
        else:
            plt.title('noise>'+str(noise[noi+1]),fontsize=textsize)
        plt.xlabel('time')
        plt.ylabel('wave')
        plt.ylim([-3,4])
        plt.legend(loc='lower right') #,fontsize=plottextsize)
    plt.subplots_adjust(left=0.07,bottom=0.08,right=0.98,top=0.95,wspace=0.3,hspace=0.35)
    plt.savefig(plotdir+'realandartificial.pdf')
    plt.show()

    
# plotartificialonly()
# plotrealandartificial()
