# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:23:29 2016

@author: jsbrown
"""

import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import estimate_bandwidth
import numpy as np
from time import time
import matplotlib.pyplot as plt

#setup some basic test data
dfpre=pd.DataFrame({'X':[0,0.1,100,5,20,21],'Y':[0,0.2,100,25,40,41],'t':'1'})
dfpst=pd.DataFrame({'X':[0,0.3,50,1,21,19,-20,20],'Y':[0,0.1,50,100,41,39,-40,40],'t':'2'})

DF=pd.concat([dfpre,dfpst])

X = np.array(DF.loc[:,['X','Y']],dtype='float')
Xpre = np.array(DF[DF['t']=='1'].loc[:,['X','Y']],dtype='float')

bw = estimate_bandwidth(X,quantile=0.3)

def plot_cluster(DF,FiltCol='t',FiltVal='2',Label='MeanShift_labels',newFig=True):
    '''
    plt.plot()
    '''
    if newFig:
        f=plt.figure(figsize=(5,5))
    else:
        f=plt.gcf()
    f.suptitle(Label.replace('_labels',''))
    plt.xlabel('X')
    plt.ylabel('Y')
    try:
        df=DF[DF[FiltCol]==FiltVal]
    except:
        df=DF
    xmin,xmax=np.min(DF['X']),np.max(DF['X'])
    ymin,ymax=np.min(DF['Y']),np.max(DF['Y'])
    dfg=df.groupby(Label)
    for n,g in dfg:
        if type(n)==str:
            plt.plot(g['X'],g['Y'],marker='o',label='Level='+n,c=plt.cm.RdBu(float(n)/len(dfg.groups)),mec='k',ls='None',markersize=5)
            leg_title='Step'
        elif n != -1:
            plt.plot(g['X'],g['Y'],marker='o',markersize=15,label=n,c=plt.cm.spectral(n/len(Xpre)),mec='k',ls='None',alpha=0.3)
            leg_title='Cluster Label'
        else:
            plt.plot(g['X'],g['Y'],marker='o',label=n,c='w',mec='r',ls='None',alpha=0.5,markersize=15)
            leg_title='Cluster Label'
    plt.legend(title=leg_title,loc=(1.1,0.5),numpoints=1,fontsize='x-small',markerscale=0.5)
    plt.xlim((xmin-5,xmax+5))
    plt.ylim((ymin-5,ymax+5))

Models={'MeanShift':MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=False),
'AffinityPropagation':AffinityPropagation(),
'KMeans_n-clusters':KMeans(n_clusters=len(dfpre),n_init=20),
'KMeans_init=pre':KMeans(n_clusters=len(dfpre),init=Xpre),
'KMeans_default':KMeans(n_init=100)}

for model in Models:
    m=Models[model]
    t0=time()
    m.fit(X)
    print('\n'+'{} complete: took {:.3f}s'.format(model,(time()-t0)).center(50,'-'))
    labels=m.labels_
    cluster_centers=m.cluster_centers_
    print('{}\n'.format(labels))
    DF[model+'_labels'] = labels
    if 'KMeans' in model:
        CentroidDistance = list(map(lambda i: pow(-m.score(X[i].reshape(1,-1)),0.5),range(0,len(X))))        
        DF[model+'_distCentroid'] = CentroidDistance

        for i,d in enumerate(CentroidDistance):
            if d > 0.5:
                labels[i]=-1
        DF[model+'_labels'] = labels
        

plot_cluster(DF,FiltCol=None,FiltVal=None,Label='t')

for model in Models:
    plot_cluster(DF,FiltCol=None,FiltVal=None,Label='t')
    plot_cluster(DF,Label=model+'_labels',newFig=False)

from sklearn.metrics.pairwise import euclidean_distances
columns=['X', 'Y', 't','KMeans_default_labels','KMeans_default_distCentroid']
df=DF.loc[:,columns]
dfg = df.groupby(['KMeans_default_labels'])
for n,g in dfg:
    print('\nLabel={}'.format(n))
    print(g)
print('\n'+'Execution Complete'.center(70,'>'))