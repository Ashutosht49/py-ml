# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:02:46 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

dataset=pd.read_csv("Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)