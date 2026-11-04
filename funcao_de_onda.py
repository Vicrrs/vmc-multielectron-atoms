#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:47:51 2026

@author: vicrrs

Exemplos resolvidos:
  1. Normalizar psi(x) = A exp(-lambda|x|) e calcular <x>, <x^2>, sigma_x
  2. Gaussiana: verificar sigma_x * sigma_p = hbar/2 (incerteza mínima)
  3. Algoritmo de Metropolis amostrando |psi|^2 (a base do VMC)
 
Unidades: hbar = m = 1.
No Spyder: rode tudo com F5, ou célula por célula com Ctrl+Enter.
Para gráficos em janela separada: Ferramentas > Preferências > IPython >
Gráficos > Backend: Automático.

"""

import numpy as np
#from matplotlib import pyplot as plt

integra = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# 1) Normalização de valores esperados de psi = A exp(-lambda|x|)
lam = 1.0
x = np.linspace(-10, 10, 20001)
psi_sem_norma = np.exp(-lam * np.abs(x))

# A² * integral(|psi|²) = 1 -> A = 1/sqrt(integral)
integral = integra(psi_sem_norma**2, x)
A = 1 / np.sqrt(integral)
psi = A * psi_sem_norma
rho = psi**2 # densidade de probabilidade

x_med = integra(x * rho, x)
x2_med = integra(x**2 )

