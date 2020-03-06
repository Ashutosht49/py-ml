# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:49:15 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset= pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#regressor=DecisionTreeRegressor(random_state=0)
#regressor.fit(x,y)
#
#y_pred=regressor.predict([[6.5]])
#
#x_grid=np.arange(min(x),max(x),0.01)
#x_grid=x_grid.reshape((len(x_grid),1))
#plt.scatter(x,y,color="red")
#plt.plot(x_grid,regressor.predict(x_grid),color="blue")
#plt.title("Truuth Vs Bluf")
#plt.xlabel("")
#plt.ylabel("")
#plt.show()

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor(n_estimators=20,random_state=0)
forest_reg.fit(x,y)
fores_pred=forest_reg.predict([[7.6]])
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,forest_reg.predict(x_grid),color="blue")
plt.title("Truuth Vs Bluf")
plt.xlabel("")
plt.ylabel("")
plt.show()