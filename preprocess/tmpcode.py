 plt.hist(dfdata[dfdata['label1hot']==0]['ene'],bins,alpha=0.7,histtype='step', label1hot='Surface',linewidth=2)


 def createArtificialnoisywaveformfromlowestnoisewave():
     
    lowestnoisewave=np.array(waveform)[datadf.noise<noise[0] ,:]
    lowestnoiselabel=np.loadtxt('../../data/dfdata/gmmlabels.txt')
    lowestnoiselabel=lowestnoiselabel.reshape(lowestnoiselabel.shape[0],1)
    lowestnoiselabel=np.concatenate((lowestnoiselabel,lowestnoiselabel),axis=0)
    for noi in range(1,7):
        noisecollection=np.loadtxt('../../data/dfdata/noisecollection'+str(noi)+'.txt')
        noisestart=np.array(random.sample(range(0,noisecollection.shape[0]-3001), lowestnoiselabel.shape[0]))
        for k in range(thiswave.shape[0]):
            thiswave1[k,:]=lowestnoisewave[k,:]+noisecollection[np.arange(noisestart[k],noisestart[k]+3000)]
        for k in range(thiswave.shape[0]):
            thiswave2[k,:]=lowestnoisewave[k,:]+noisecollection[np.arange(noisestart[k],noisestart[k]+3000)]
        thiswave=np.concatenate((thiswave1,thiswave2),axis=0)
        thiswave=np.concatenate((thiswave,lowestnoiselabel),axis=1)
        np.savetxt('../../data/dfdata/artificial'+str(noi)+'.txt',thiswave)
 



def extractfeaturefromkmeansdata(kmeanswave):
    length95=np.abs(kmeanswave-0.95).argmin(axis=1)-np.abs(kmeanswave-0.05).argmin(axis=1)
    length90=np.abs(kmeanswave-0.9).argmin(axis=1)-np.abs(kmeanswave-0.1).argmin(axis=1)
length85=np.abs(kmeanswave-0.85).argmin(axis=1)-np.abs(kmeanswave-0.15).argmin(axis=1)
length80=np.abs(kmeanswave-0.8).argmin(axis=1)-np.abs(kmeanswave-0.2).argmin(axis=1)

length75=np.abs(kmeanswave-0.75).argmin(axis=1)-np.abs(kmeanswave-0.25).argmin(axis=1)
length70=np.abs(kmeanswave-0.7).argmin(axis=1)-np.abs(kmeanswave-0.3).argmin(axis=1)
length65=np.abs(kmeanswave-0.65).argmin(axis=1)-np.abs(kmeanswave-0.35).argmin(axis=1)
length60=np.abs(kmeanswave-0.6).argmin(axis=1)-np.abs(kmeanswave-0.4).argmin(axis=1)
length55=np.abs(kmeanswave-0.55).argmin(axis=1)-np.abs(kmeanswave-0.45).argmin(axis=1)

kmeansfeature=np.concatenate((length95,length90,length85,length80,length75,length70,length65,length60,length55),axis=0).reshape(9,length90.shape[0]).T
tau=np.log(19)/np.array(datadf[datadf.noise<kmeansthre]['fitslope']).reshape(len(length90),1)
ampli=np.array(datadf[datadf.noise<kmeansthre]['ampli']).reshape(len(length90),1)
data=np.concatenate((kmeansfeature,tau,ampli),axis=1)
plt.hist(kmeansfeature[:,0],bins=np.linspace(0,400,150))
 plt.hist(kmeansfeature[:,1],bins=np.linspace(0,250,50))
plt.hist(kmeansfeature[:,3],bins=np.linspace(0,80,50))

plt.show()

    
