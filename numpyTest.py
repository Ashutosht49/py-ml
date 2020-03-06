# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:23:42 2019

@author: Administrator
"""

import numpy as np



def pypart(n): 
    myList = [] 
    for i in range(1,n+1): 
        myList.append("*"*i) 
    print("\n".join(myList)) 
  
# Driver Code 
n = 5
pypart(n) 