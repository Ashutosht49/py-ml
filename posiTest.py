# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:52:39 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

salaryDataset=pd.read_csv("Position_Salaries.csv")
x=salaryDataset.iloc[:,1].values
y=salaryDataset.iloc[:,2].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape=(-1,1)
x_test.shape=(-1,1)

sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

regr=LinearRegression()
regr.fit(x_train,x_test)
y_pred=regr.predict(x_test)

#plt.scatter(x,y,color="red")
#plt.xlabel("Experiance")
#plt.ylabel("Salary")

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)
x=x.reshape(-1,1)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y_train)
#lin_reg2=LinearRegression()
regr.fit(x_poly,y_train)

plt.scatter(x_train,y_train,color="green")
#plt.plot(x_train,regr.predict(x_poly),color="blue")
plt.xlabel("Experiance")
plt.ylabel("Salary")





