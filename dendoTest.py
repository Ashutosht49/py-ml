# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:06:35 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset=pd.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,[3,4]].values

#dendogram=sch.dendrogram(sch.linkage(x,method="ward"))
#plt.title("Dendogram")
#plt.xlabel("Cluster")
#plt.ylabel("Eculiance Diagram")
#plt.show()


hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_kmeans=hc.fit_predict(x)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Cluter1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='Cluter2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Cluter3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='cyan',label='Cluter4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='magenta',label='Cluter5')

