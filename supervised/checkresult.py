import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from array import array
import pandas as pd

textsize=15
matplotlib.rcParams.update({'font.size': textsize})
supdir='../../data/supervised/'
unsupdir='../../data/unsupervised/'
suptxtdir='../../txt/supervised/'
source="Cs137" #"Co60"

thre=[0.5,0.5,0.5,0.5,0.85,0.99,1.2]

paradf = pd.DataFrame(columns=['idev','ampli','ped','cross','slope','tau','noise','glabels','label1hot','noi'])
paradflist=[]
for noi in range(0,7):
    if noi==0:
        with open(unsupdir+source+'testresultlabel.dat', 'rb') as f:
            para=pickle.load(f)
        para['label1hot']=1-(para['glabels']>thre[noi]).astype('int64')
        para['noi']=list(map(int,np.ones(para.shape[0])*noi))
        paradflist.append(para)
        paradf=paradf.append(para)
    else:
        with open(supdir+source+'predlabel'+str(noi)+'.dat', 'rb') as f:
            para=pickle.load(f)
        para['label1hot']=(para['glabels']>thre[noi]).astype('int64')
        para['noi']=list(map(int,np.ones(para.shape[0])*noi))
        paradflist.append(para)
        paradf=paradf.append(para)
    print(noi,para.shape,paradf.shape)
# paradf.reindex()
paradf['ene']=paradf['ampli']*1.334*1e-3+5.8766*1e-2


# fig = plt.figure(figsize=(8,7))
# for noi in range(1,7):
#     ax= plt.subplot2grid((2,3), ((noi-1)//3,(noi-1)%3))
#     ax.hist(paradflist[noi].glabels,bins=np.linspace(-0.5,1.5,50))
#     plt.title('noi='+str(noi))
# plt.show()

# bins=np.linspace(-1,2,50)
# # fig = plt.figure(figsize=(8,7))
# # plt.scatter(paradf.ene,paradf.tau)
# paradf.plot.scatter(x='ene',y='tau',c='glabels',colormap='viridis',s=0.5)
# plt.show()

# bins=np.linspace(0,5,50)
# paradf[paradf['noi']==0]['glabels'].hist(bins=bins)
# plt.show()
fig = plt.figure(figsize=(8,7))
bins = np.linspace(0, 5,70)
plt.hist(paradf['ene'],bins,alpha=1,histtype='step', label='Full',linewidth=2) #,log=True)
plt.hist(paradf[paradf['label1hot']==1]['ene'],bins,alpha=1,histtype='step', label='Bulk',linewidth=2)
plt.hist(paradf[paradf['label1hot']==0]['ene'],bins,alpha=1,histtype='step', label='Surface',linewidth=2)

plt.xlabel("energy [keVee]",fontsize=textsize)
plt.ylabel("Counts",fontsize=textsize)
plt.legend(loc='upper right',fontsize=textsize)
plt.xlim([0, 5])
plt.ylim([0, 200])
# plt.savefig(model+"energyregsmall.pdf",dpi=200)
# plt.savefig(model+"energyreg.pdf",dpi=200)
plt.show()


bins = np.linspace(0.16,11.66,231)
a=plt.hist(paradf[paradf['label1hot']==1]['ene'],bins)
bulkdf=pd.DataFrame({'energy':a[1][:-1]+0.025,'count':a[0]})
bulkdf.to_csv(suptxtdir+'bulk.csv')
