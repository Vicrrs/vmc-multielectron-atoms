#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
sigma_x = np.sqrt(x2_med - x_med**2)

print("=== 1) psi = A exp(-lambda|x|), lambda = 1 ===")
print(f"A numérico  = {A:.6f}   A exato = sqrt(lambda) = {np.sqrt(lam):.6f}")
print(f"<x>         = {x_med:.6f}   (exato 0, pela simetria)")
print(f"<x^2>       = {x2_med:.6f}   exato 1/(2 lambda^2) = {1/(2*lam**2):.6f}")
print(f"sigma_x     = {sigma_x:.6f}   exato = {1/(np.sqrt(2)*lam):.6f}")


# probabilidade de encontrar a partícula a mais de 1 sigma da origem
fora = np.abs(x) > sigma_x
P_fora = integra(np.where(fora, rho, 0), x )
print(f"P(|x| > sigma) = {P_fora:.4f}   exato exp(-sqrt(2)) = {np.exp(-np.sqrt(2)):.4f}\n")


# Gaussiana princípio da incerteza
a = 0.7
psi_g = (2 * a / np.pi) ** 0.25 * np.exp(-a * x**2) #normalizado
dpsi = np.gradient(psi_g, x)


sx = np.sqrt(integra(x**2 * psi_g**2, x))
# <p^2> = integral |dpsi/dx|^2 dx   (integração por partes, hbar = 1)

sp = np.sqrt(integra(dpsi**2, x))

print("=== 2) Gaussiana ===")
print(f"sigma_x = {sx:.5f}, sigma_p = {sp:.5f}, produto = {sx*sp:.5f}  (hbar/2 = 0.5)")

# comparar com a exponencial do item 1 (não é incerteza mínima)
sp_exp = np.sqrt(integra(np.gradient(psi, x)**2, x))
print(f"exponencial: sigma_x*sigma_p = {sigma_x*sp_exp:.5f}  (> 0.5)\n")











