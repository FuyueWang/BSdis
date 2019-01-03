from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture as GMM
import pickle
import matplotlib
import matplotlib.pyplot as plt
textsize=15
matplotlib.rcParams.update({'font.size': textsize})
plotdir='../../plot/unsupervised/'
datadir='../../data/unsupervised/'
preprossdatadir='../../data/preprocess/'
source="Cs137" #"Co60"

with open(datadir+source+'featuretrain.dat', 'rb') as f:
    feature = pickle.load(f)
    
X=feature[:,4:]

gmm = GMM(n_components=2,covariance_type='full', max_iter=100, random_state=20).fit(X)
glabels = gmm.predict(X)

kmeans = KMeans(n_clusters=2, n_init=20).fit(X)
klabels = kmeans.predict(X)
          
density = DBSCAN(eps=0.5,min_samples=10).fit(X)
dlabels = density.labels_

# save glabel result
with open(preprossdatadir+source+'normedwaveform0.dat', 'rb') as f:
    data = pickle.load(f)
paradf=data['para']
paradf['glabels']=glabels
with open(datadir+source+'testresultlabel.dat', 'wb') as f:
    pickle.dump(paradf,f)


fig = plt.figure(figsize=(15,7))
ax = plt.subplot2grid((1,3), (0,0))  
ax.scatter(data['para'].ampli,data['para'].tau,c=klabels,s=0.5)
plt.title('kMeans')
plt.xlabel('Amplitude 5% - 95%')
plt.ylabel('Rising time')
ax = plt.subplot2grid((1,3), (0,1))  
ax.scatter(data['para'].ampli,data['para'].tau,c=glabels,s=0.5)
plt.title('GMM')
plt.xlabel('Amplitude 5% - 95%')
plt.ylabel('Rising time')
ax = plt.subplot2grid((1,3), (0,2))  
ax.scatter(data['para'].ampli,data['para'].tau,c=dlabels,s=0.5)
plt.title('Density',fontsize=textsize)
plt.xlabel('Amplitude 5% - 95%')
plt.ylabel('Rising time')
plt.subplots_adjust(left=0.06,bottom=0.08,right=0.98,top=0.95,wspace=0.25,hspace=0.35)
plt.savefig(plotdir+'unsupervised.png',dpi=200)
# plt.savefig(plotdir+'unsupervised.pdf')
plt.show()
