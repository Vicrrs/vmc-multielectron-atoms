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
from matplotlib import pyplot as plt

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


# 2) Gaussiana princípio da incerteza
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

# 3) Metropolis: sortear x com probabilidade |psi|^2 sem conhecer A
rng = np.random.default_rng(1)

def rho_sem_norma(x):
    return np.exp(-2 * lam * np.abs(x)) # |psi|^2 com a constante

N, passo = 200_000, 1.5
amostras = np.empty(N)
xa, aceitos = 0.0, 0
for k in range(N):
    xn = xa + passo * rng.uniform(-1, 1) # proposta
    # aceita com probabilidade min(1, rho(xn)/rho(xa)): a constante A se cancela
    if rng.random() < rho_sem_norma(xn) / rho_sem_norma(xa):
        xa = xn
        aceitos += 1
    amostras[k] = xa
amostras = amostras[2000:] # descartando o aquecimento

x2_mc = np.mean(amostras**2)
# erro por bloco (amostras vizinhas sao correlacionadas)
blocos = amostras[: len(amostras) // 100 * 100].reshape(100, -1)
erro = np.std(np.mean(blocos**2, axis=1), ddof=1) / np.sqrt(100)
print("=== 3) Metropolis ===")
print(f"aceitação = {aceitos/N:.2f}")
print(f"<x^2> Monte Carlo = {x2_mc:.4f} +- {erro:.4f}   exato = {1/(2*lam**2):.4f}")


# Gráficos
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
 
ax[0].plot(x, rho, "C0", lw=2, label=r"$|\psi|^2$")
ax[0].fill_between(x, rho, where=fora, color="C3", alpha=0.3,
                   label=fr"$|x|>\sigma$: P = {P_fora:.3f}")
ax[0].axvline(-sigma_x, color="gray", ls="--")
ax[0].axvline(sigma_x, color="gray", ls="--")
ax[0].set(xlim=(-4, 4), xlabel="x", ylabel="densidade",
          title=r"1) $\psi = A\,e^{-\lambda|x|}$ normalizada")
ax[0].legend()
 
ax[1].plot(x, psi_g**2, "C0", lw=2, label=r"$|\psi(x)|^2$ gaussiana")
ax[1].plot(x, psi**2, "C1", lw=2, label=r"$|\psi(x)|^2$ exponencial")
ax[1].set(xlim=(-4, 4), xlabel="x",
          title=fr"2) $\sigma_x\sigma_p$: gauss {sx*sp:.3f}, exp {sigma_x*sp_exp:.3f}")
ax[1].legend()
 
ax[2].hist(amostras, bins=120, range=(-5, 5), density=True, color="C2",
           alpha=0.6, label="Metropolis")
ax[2].plot(x, rho, "k", lw=1.5, label=r"$|\psi|^2$ exato")
ax[2].set(xlim=(-5, 5), xlabel="x", title="3) Amostrando $|\\psi|^2$ com Metropolis")
ax[2].legend()
 
fig.tight_layout()
plt.show()
