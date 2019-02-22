# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:06:43 2019

@author: Usuario
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer,SigmoidLayer,BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()
camadaEntrada = LinearLayer(25)
camadaOculta = SigmoidLayer(26)
camadaSaida = SigmoidLayer(1)

bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addInputModule(bias1)
rede.addInputModule(bias2)


entradaOculta = FullConnection(camadaEntrada,camadaOculta)
ocultaSaida = FullConnection(camadaOculta,camadaSaida)

biasOculta = FullConnection(bias1,camadaOculta)
biasSaida = FullConnection(bias2,camadaSaida)

rede.sortModules()


print(rede)
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)

