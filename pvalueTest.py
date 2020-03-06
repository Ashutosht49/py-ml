# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:08:37 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

salaryDataset=pd.read_csv("50_Startups.csv")
x=salaryDataset.iloc[:,:4].values
y=salaryDataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableEncoder_x=LabelEncoder()
x[:,3]=lableEncoder_x.fit_transform(x[:,3])
oneHotEncoder_x=OneHotEncoder(categorical_features=[3])
x=oneHotEncoder_x.fit_transform(x).toarray()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regr=LinearRegression()
regr.fit(x_train,y_train)

y_pred=regr.predict(x_test)

#plt.plot(np.arange(10),y_test,color="red")
#plt.plot(np.arange(10),regr.predict(x_test),color="blue")
#plt.xlabel("Experiance")
#plt.ylabel("Salary")

import statsmodels.api as sm
a=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
a_opt=a[:,[0,1,2,3,4,5]]
regre_OLS=sm.OLS(endog=y,exog=a_opt)
regre_OLS=regre_OLS.fit()
regre_OLS.summary()
