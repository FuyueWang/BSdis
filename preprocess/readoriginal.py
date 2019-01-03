import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from array import array
import pandas as pd
import os.path
textsize=15
matplotlib.rcParams.update({'font.size': textsize})
originaldatadir='../../data/originaldata/'
outdatadir='../../data/preprocess/'
plotdir='../../plot/preprocess/'
unsuperviseddatadir='../../data/unsupervised/'

source="Cs137" #"Co60"

noise=[0,0.025,0.05,0.07,0.1,0.2,0.4]


def splitdatabynoise():
    Nbofnoisepoint=200
    infile = open(originaldatadir+'idevt_pulse_reduced_'+source+'.txt', 'rb')
    Nboflines = 62438 #Co60-71966 Cs137-62438 #len(infile.readlines())

    listparadf=[pd.DataFrame(columns=['idev','ampli','ped','cross','slope','tau','noise']) for k in range(7)]
    waveformlist=[[] for k in range(7)]
    noisecollectionlist=[[] for k in range(7)]

    # Nboflines=10
    for i in range(Nboflines):
        linedata=list(map(float,infile.readline().split()))    
        startval=np.mean(linedata[5:5+Nbofnoisepoint])
        endval=np.mean(linedata[-Nbofnoisepoint:])
        waveform=(np.array(linedata[5:])-startval)/(endval-startval)
        noisedata=np.concatenate((waveform[5:5+Nbofnoisepoint],waveform[-Nbofnoisepoint:]-1),axis=0)
        noiseval=np.std(noisedata)
        noiseid=np.concatenate((noiseval>np.array(noise),np.array([False])),axis=0).argmin()-1

        if waveform.shape[0]!=3000:
            print(i,waveform.shape)
        waveformlist[noiseid].append(waveform[:3000])
        noisecollectionlist[noiseid]+=list(noisedata)
        listparadf[noiseid].loc[listparadf[noiseid].shape[0]]=linedata[0:5]+[np.log(19)/linedata[4]]+[noiseval]
    
    for noi in range(7):
        with open(outdatadir+source+'noisecollection'+str(noi)+'.dat', 'wb') as f:
            pickle.dump(np.array(noisecollectionlist[noi]), f)
        dictdata = dict(waveform=np.array(waveformlist[noi]),para=listparadf[noi])
        with open(outdatadir+source+'normedwaveform'+str(noi)+'.dat', 'wb') as f:
            pickle.dump(dictdata, f)
    
# splitdatabynoise()

def plotnoisecollection():
    fig = plt.figure(figsize=(10,7))
    for i in range(6):
        Nbofplotnoisepoint=5000
        with open(outdatadir+source+'noisecollection'+str(i+1)+'.dat', 'rb') as f:
            data = pickle.load(f)
        startid=int(np.round(np.random.uniform(0,len(data)-Nbofplotnoisepoint),0))
        ax = plt.subplot2grid((2,3), (i//3,i%3))
        ax.plot(data[startid:startid+Nbofplotnoisepoint])
        if i <5:
            plt.title(str(noise[i+1])+'<noise<'+str(noise[i+2]),fontsize=textsize)
        else:
            plt.title('noise>'+str(noise[i+1]),fontsize=textsize)
        plt.xlabel('time')
        plt.ylabel('noise')
        plt.ylim([-3,3])
    plt.subplots_adjust(left=0.07,bottom=0.08,right=0.98,top=0.95,wspace=0.3,hspace=0.35)
    plt.savefig(plotdir+'noisecollection.pdf')
    plt.show()

# plotnoisecollection()

def plotAforeachnoise(rangeflag='Small',xlabelflag="Amplitude"):
    fig = plt.figure(figsize=(8,7))
    if xlabelflag=="Amplitude" and rangeflag=='Small':
        bins= np.linspace(0,4000,100)
    elif xlabelflag=="Amplitude" and rangeflag=='Large':
        bins= np.linspace(0,15000,100)
    elif xlabelflag=="Energy" and rangeflag=='Small':
        bins= np.linspace(0,5.5,100)
    elif xlabelflag=="Energy" and rangeflag=='Large':
        bins= np.linspace(0,20,100)
        
    for i in range(7):
        with open(outdatadir+source+'normedwaveform'+str(i)+'.dat', 'rb') as f:
            data = pickle.load(f)
        if xlabelflag=="Amplitude":
            if i==0:
                plt.hist(data['para'].ampli,bins=bins,label=str('noise<'+str(noise[i+1])),histtype='step',linewidth=2)
            elif i==6:
                plt.hist(data['para'].ampli,bins=bins,label=str('noise>'+str(noise[i])),histtype='step',linewidth=2)
            else:
                plt.hist(data['para'].ampli,bins=bins,label=str(noise[i])+'<noise<'+str(noise[i+1]),histtype='step',linewidth=2)
        else:
            if i==0:
                plt.hist(data['para'].ampli*1.334*1e-3+5.8766*1e-2,bins=bins,label=str('noise<'+str(noise[i+1])),histtype='step',linewidth=2)
            elif i==6:
                plt.hist(data['para'].ampli*1.334*1e-3+5.8766*1e-2,bins=bins,label=str('noise>'+str(noise[i])),histtype='step',linewidth=2)
            else:
                plt.hist(data['para'].ampli*1.334*1e-3+5.8766*1e-2,bins=bins,label=str(noise[i])+'<noise<'+str(noise[i+1]),histtype='step',linewidth=2)            
    plt.subplots_adjust(left=0.12,bottom=0.09,right=0.98,top=0.95,wspace=0.3,hspace=0.35)
    if xlabelflag=="Amplitude":
        plt.xlabel(xlabelflag)
    else:
        plt.xlabel(xlabelflag+' [keVee]')
    plt.ylabel('Count')
    plt.legend(loc='upper right') #,fontsize=plottextsize)
    plt.savefig(plotdir+xlabelflag+'histwithnoise'+rangeflag+'.pdf')
    # plt.savefig(plotdir+'Ahistwithnoise.pdf')
    plt.show()

# plotAforeachnoise()
# plotAforeachnoise(rangeflag='Large',xlabelflag='Energy')

def createunsuperviseddata():
    print('createunsuperviseddata')
    with open(outdatadir+source+'normedwaveform0.dat', 'rb') as f:
        data = pickle.load(f)
    wave=data['waveform']
    length95=np.abs(wave-0.95).argmin(axis=1)-np.abs(wave-0.05).argmin(axis=1)
    length90=np.abs(wave-0.9).argmin(axis=1)-np.abs(wave-0.1).argmin(axis=1)
    length85=np.abs(wave-0.85).argmin(axis=1)-np.abs(wave-0.15).argmin(axis=1)
    length80=np.abs(wave-0.8).argmin(axis=1)-np.abs(wave-0.2).argmin(axis=1)
    length75=np.abs(wave-0.75).argmin(axis=1)-np.abs(wave-0.25).argmin(axis=1)
    length70=np.abs(wave-0.7).argmin(axis=1)-np.abs(wave-0.3).argmin(axis=1)
    length65=np.abs(wave-0.65).argmin(axis=1)-np.abs(wave-0.35).argmin(axis=1)
    length60=np.abs(wave-0.6).argmin(axis=1)-np.abs(wave-0.4).argmin(axis=1)
    length55=np.abs(wave-0.55).argmin(axis=1)-np.abs(wave-0.45).argmin(axis=1)
    feature=np.concatenate((length95,length90,length85,length80,length75,length70,length65,length60,length55),axis=0).reshape(9,length90.shape[0]).T
    print('feature shape:',feature.shape)
    with open(unsuperviseddatadir+source+'featuretrain.dat', 'wb') as f:
        pickle.dump(feature, f)

createunsuperviseddata()

