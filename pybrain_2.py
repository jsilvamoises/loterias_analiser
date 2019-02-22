# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:45:16 2019

@author: Usuario
"""
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer
import pandas as pd
import numpy as np
"""
opt = {'bias': True,
           'hiddenclass': SigmoidLayer,
           'outclass': LinearLayer,
           'outputbias': True,
           'peepholes': False,
           'recurrent': False,
           'fast': False,
    }
"""
# rede = buildNetwork(25,26,1,outclass= SoftmaxLayer,hiddenclass=SigmoidLayer,bias=False)

"""
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])
"""
rede = buildNetwork(25,26,1)# (camada de entrada,camadaIntermediaria,camadaSaida)
base = SupervisedDataSet(25,1)

dados = pd.read_csv('lotofacil.csv',sep=';')

dados_treinamento = dados.iloc[:1200,0:25]
previsores_treinamento = dados.iloc[:1200,25:]

dados_validacao = dados.iloc[1201:,0:25]
previsores_validacao = dados.iloc[1201:,25:]

base.setField('input',dados_treinamento)
base.setField('target',previsores_treinamento)

print(base['input'])
print(base['target'])

treinamento = BackpropTrainer(rede,dataset=base,learningrate=0.05,momentum=0.03)

for i in range(1,1000):
    erro = treinamento.train()
    print('Epoca %d erro: %.10f'%(i,erro))
    if i % 1000 == 0:
        print('a')
        
nao_sorteados = dados[dados['ST']==0] 
nao_sorteados = nao_sorteados.iloc[:,0:25]    
nao_sorteados = np.array(nao_sorteados)


sorteados = dados[dados['ST'] > 0] 
sorteados = dados.iloc[:,0:25]  
sorteados = np.array(sorteados)  

for ns in sorteados:
    print(rede.activate(ns))

for ns in nao_sorteados:
    print(rede.activate(ns))


print(nao_sorteados[0])



print(rede.activate([0,2,3,4,0,6,0,8,9,0,0,0,0,0,15,0,0,18,19,20,21,22,23,24,25]))
print(rede.activate([1,0,3,4,0,0,7,0,0,10,0,0,13,14,15,0,17,18,0,20,21,22,23,0,25]))
print(rede.activate([1,0,0,0,5,6,0,0,0,10,11,12,13,0,15,0,17,18,19,20,21,0,23,24,0]))


### 1
print('1')
print(rede.activate([1,2,3,0,0,6,0,0,0,10,0,12,13,14,15,0,0,0,19,20,21,0,23,24,25]))
print(rede.activate([0,0,3,0,0,6,7,0,9,0,11,12,13,0,15,16,0,18,19,0,0,22,23,24,25]))
print(rede.activate([0,2,0,4,5,6,0,0,0,10,11,12,0,0,15,16,0,18,19,20,0,22,0,24,25]))
print(rede.activate([1,0,3,0,0,6,0,0,9,10,0,0,13,14,0,0,17,18,19,20,21,22,23,24,0]))

print('0')
### 0
print(rede.activate([1,0,0,0,5,6,0,0,0,10,11,12,13,0,15,0,17,18,19,20,21,0,23,24,0]))
print(rede.activate([1,0,3,4,0,0,7,0,0,10,0,0,13,14,15,0,17,18,0,20,21,22,23,0,25]))
print(rede.activate([1,0,0,0,0,0,7,8,0,10,11,12,0,14,15,16,0,0,19,20,0,22,23,24,25]))
print(rede.activate([0,2,3,4,5,0,7,0,0,0,11,12,0,14,15,0,0,18,0,0,21,22,23,24,25]))
        
        
    
