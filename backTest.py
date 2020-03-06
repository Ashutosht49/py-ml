# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:39:06 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm

def backwordElimination(x, sl):
    numVars=len(x[0])
    for i in range(0, numVars):
        regre_OLS=sm.OLS(y,x).fit