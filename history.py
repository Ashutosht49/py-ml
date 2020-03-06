# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Wed Dec  4 10:41:58 2019)---
runfile('C:/Users/Administrator/.spyder-py3/temp.py', wdir='C:/Users/Administrator/.spyder-py3')
a="Hello Word"
for index, value in enumerate(a):
    print(index, value)
    break

else:
    print("Success")
    
    print(index, value)
runfile('C:/Users/Administrator/.spyder-py3/temp.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/numpyTest.py', wdir='C:/Users/Administrator/.spyder-py3')
def pypart(n):
debugfile('C:/Users/Administrator/.spyder-py3/numpyTest.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/pandaTest.py', wdir='C:/Users/Administrator/.spyder-py3')

## ---(Thu Dec  5 10:31:58 2019)---
runfile('C:/Users/Administrator/.spyder-py3/pandaTest.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/pvalueTest.py', wdir='C:/Users/Administrator/.spyder-py3')
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
import statsmodels.api as sm
a=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
a_opt=a[:,[0,1,2,3,4,5]]
regre_OLS=sm.OLS(endog=y,exog=a_opt)
regre_OLS=regre_OLS.fit()
regre_OLS.summary()
runfile('C:/Users/Administrator/.spyder-py3/posiTest.py', wdir='C:/Users/Administrator/.spyder-py3')

## ---(Fri Dec  6 10:48:56 2019)---
runfile('C:/Users/Administrator/.spyder-py3/treeTest.py', wdir='C:/Users/Administrator/.spyder-py3')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset= pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1].values
y=dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("Truuth Vs Bluf")
plt.xlabel("")
plt.ylabel("")
plt.show()
runfile('C:/Users/Administrator/.spyder-py3/treeTest.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/Test1.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/knTest.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/test2.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/test4.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/mallCust.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/dendoTest.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/bayesTest.py', wdir='C:/Users/Administrator/.spyder-py3')
runfile('C:/Users/Administrator/.spyder-py3/test5.py', wdir='C:/Users/Administrator/.spyder-py3')