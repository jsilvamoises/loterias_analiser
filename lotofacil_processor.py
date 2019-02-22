# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 02:34:45 2019
https://www.udemy.com/machine-learning-e-data-science-com-python-y/learn/v4/t/lecture/10203250?start=0
@author: Usuario
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,StandardScaler
from pybrain.structure import FeedForwardNetwork


base = pd.read_csv('lotofacil.csv',sep=';')

previsores = base.iloc[:,0:25]
classe = base.iloc[:,25:]

print(base.describe())

treinamento = base.copy()
treinamento = treinamento.drop('ST',1)
treinamento = treinamento/25
expected = base['ST']

print(treinamento['D25'].mean())
print(expected.mean())


