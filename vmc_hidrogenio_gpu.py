#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:27:21 2026

@author: roza
"""

# -*- coding: utf-8 -*-
"""
VMC do átomo de hidrogênio na GPU - Python + Numba CUDA
========================================================
O primeiro código "de projeto": mesmo esquema que vamos usar para He, Li, Be e B,
só que com 1 elétron, para a física e o CUDA ficarem fáceis de acompanhar.

A FÍSICA (unidades atômicas: energia em hartree, distância em bohr)
  Função de onda tentativa:   psi(r) = exp(-alpha * r)
  Hamiltoniano:               H = -1/2 laplaciano - 1/r
  Energia local:              E_L = H psi / psi = -alpha^2/2 + (alpha - 1)/r
  Energia variacional:        E(alpha) = media de E_L = alpha^2/2 - alpha
  O mínimo é alpha = 1, com E = -0.5 Ha (o valor exato do hidrogênio).
  Em alpha = 1, E_L = -0.5 em TODO ponto: a variância é zero.

O ALGORITMO (Metropolis)
  Cada walker é um ponto (x, y, z) que passeia pelo espaço:
    1. propõe um passo aleatório:  R' = R + passo * (número entre -1 e 1)
    2. aceita com probabilidade |psi(R')/psi(R)|^2 = exp(-2 alpha (r' - r))
    3. depois do aquecimento, soma E_L(R) a cada passo
  Depois de muitos passos, os pontos visitados seguem |psi|^2, e a média de E_L
  é a energia.

O CUDA
  1 thread = 1 walker. Cada thread faz a sua cadeia inteira sozinha e,
  no fim, escreve a média de E_L dela. A CPU tira a média entre os walkers.
  O gerador de números aleatórios (xorshift64*) é escrito à mão, igual nas
  versões C++ e Fortran: com a mesma semente, as três dão o MESMO resultado.

Rodar no Spyder com F5. Sem GPU NVIDIA, deixe a linha do simulador ativa.
"""
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"    # simulador (sem GPU). Com GPU NVIDIA, apague esta linha.
os.environ["NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING"] = "1"

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, uint64

SIMULADOR = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
U = np.uint64          # inteiro de 64 bits sem sinal (o gerador precisa dele)


# ---------------------------------------------------------------------------
# 1) Gerador de números aleatórios, um por thread
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def semente_inicial(z):
    """splitmix64: transforma (semente + número do walker) num estado bem embaralhado."""
    z = z + U(0x9E3779B97F4A7C15)
    z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
    return z ^ (z >> U(31))


@cuda.jit(device=True)
def aleatorio(estado):
    """xorshift64*: devolve um número em [0, 1) e atualiza estado[0]."""
    s = estado[0]
    s ^= s >> U(12)
    s ^= s << U(25)
    s ^= s >> U(27)
    estado[0] = s
    return float((s * U(0x2545F4914F6CDD1D)) >> U(11)) * (1.0 / 9007199254740992.0)


# ---------------------------------------------------------------------------
# 2) O kernel: cada thread é um walker
# ---------------------------------------------------------------------------
@cuda.jit
def vmc_kernel(alpha, passos, aquec, passo, semente, E_walker, aceitos_walker):
    w = cuda.grid(1)                       # número deste walker
    if w >= E_walker.size:
        return

    estado = cuda.local.array(1, dtype=uint64)
    estado[0] = semente_inicial(U(semente) + U(w))

    # posição inicial: um ponto qualquer no cubo [-1, 1]^3
    x = 2.0 * aleatorio(estado) - 1.0
    y = 2.0 * aleatorio(estado) - 1.0
    z = 2.0 * aleatorio(estado) - 1.0
    r = math.sqrt(x * x + y * y + z * z)

    soma_E = 0.0
    aceitos = 0
    for p in range(passos):
        # 1. proposta
        xn = x + passo * (2.0 * aleatorio(estado) - 1.0)
        yn = y + passo * (2.0 * aleatorio(estado) - 1.0)
        zn = z + passo * (2.0 * aleatorio(estado) - 1.0)
        rn = math.sqrt(xn * xn + yn * yn + zn * zn)

        # 2. aceitação: compara log(u) com log de |psi(R')/psi(R)|^2 = -2 alpha (r' - r)
        u = aleatorio(estado)
        if u > 0.0 and math.log(u) < -2.0 * alpha * (rn - r):
            x, y, z, r = xn, yn, zn, rn
            aceitos += 1

        # 3. energia local (só depois do aquecimento)
        if p >= aquec:
            soma_E += -0.5 * alpha * alpha + (alpha - 1.0) / r

    E_walker[w] = soma_E / (passos - aquec)
    aceitos_walker[w] = aceitos / passos


# ---------------------------------------------------------------------------
# 3) A CPU: lança o kernel para cada alpha e junta os resultados
# ---------------------------------------------------------------------------
W = 64 if SIMULADOR else 65536            # número de walkers (threads)
PASSOS = 300 if SIMULADOR else 2000
AQUEC = PASSOS // 10
PASSO = 1.0
TPB = 128                                 # threads por bloco
BLOCOS = (W + TPB - 1) // TPB

alphas = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
d_E = cuda.device_array(W)                # resultados ficam na GPU até o fim
d_ac = cuda.device_array(W)

E_med, E_err = [], []
print(f"{W} walkers x {PASSOS} passos (aquecimento {AQUEC})\n")
print(" alpha   E (VMC)                exato      aceitação")
for alpha in alphas:
    vmc_kernel[BLOCOS, TPB](alpha, PASSOS, AQUEC, PASSO, 12345, d_E, d_ac)
    E = d_E.copy_to_host()
    ac = d_ac.copy_to_host()
    m = E.mean()
    erro = E.std(ddof=1) / math.sqrt(W)   # walkers independentes
    E_med.append(m)
    E_err.append(erro)
    print(f" {alpha:.2f}   {m:.6f} +- {erro:.6f}   {alpha**2 / 2 - alpha:.6f}   {ac.mean():.2f}")

# ---------------------------------------------------------------------------
# 4) Gráfico
# ---------------------------------------------------------------------------
a = np.linspace(0.5, 1.5, 200)
plt.plot(a, a**2 / 2 - a, "k", label=r"exato: $\alpha^2/2 - \alpha$")
plt.errorbar(alphas, E_med, E_err, fmt="o", label="VMC na GPU")
plt.axhline(-0.5, color="gray", ls=":")
plt.xlabel(r"$\alpha$")
plt.ylabel("E (hartree)")
plt.title("Hidrogênio: energia variacional (mínimo em alpha = 1)")
plt.legend()
plt.show()

# EXERCÍCIOS
#  1. Em alpha = 1 a barra de erro é zero. Explique pela fórmula de E_L.
#  2. Mude PASSO para 0.2 e para 4.0. O que acontece com a aceitação? E com o erro?
#  3. Calcule também <r> (some r em vez de E_L). O exato é 3/(2 alpha).
#  4. Próximo passo do projeto: hélio (2 elétrons, 6 coordenadas e o termo 1/r12).