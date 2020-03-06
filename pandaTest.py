# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:26:45 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

#df=pd.read_csv("Salary_Data.csv")
#print(df)

#df=pd.read_csv("Position_Salaries.csv")
#print(df)

dataset=pd.read_csv("Data.csv")
x=dataset.iloc[:,:3].values
y=dataset.iloc[:,3].values
imputer=Imputer(missing_values=np.nan, strategy="mean")
imputer=imputer.fit(x[:,1:])
x[:,1:]=imputer.transform(x[:,1:])
#print(dataset)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableEncoder_x=LabelEncoder()
x[:,0]=lableEncoder_x.fit_transform(x[:,0])
oneHotEncoder_x=OneHotEncoder(categorical_features=[0])
x=oneHotEncoder_x.fit_transform(x).toarray()
lableEncoder_y=LabelEncoder()
y=lableEncoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
print(y)


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

salaryDataset=pd.read_csv("Salary_Data.csv")
x=salaryDataset.iloc[:,0].values
y=salaryDataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regr=LinearRegression()
x_train.shape=(-1,1)
y_train.shape=(-1,1)
regr.fit(x_train,y_train)
x_test.shape=(-1,1)
y_pred=regr.predict(x_test)

y_pred_s=regr.predict([[6.5]])

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regr.predict(x_train),color="blue")
plt.xlabel("Experiance")
plt.ylabel("Salary")

print(x)