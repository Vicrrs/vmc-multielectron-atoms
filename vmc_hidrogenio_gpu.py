# -*- coding: utf-8 -*-
"""
VMC do átomo de hidrogênio na GPU - versão Python (Numba CUDA)
================================================================
O QUE O PROGRAMA CALCULA
  A energia do hidrogênio usando a função tentativa  psi(r) = exp(-alpha * r).
  Para cada alpha, o resultado deve ser  E(alpha) = alpha^2/2 - alpha  (hartree),
  com mínimo E = -0.5 em alpha = 1 (a função exata).

AS TRÊS IDEIAS DE FÍSICA
  1) Metropolis: o elétron anda ao acaso; cada passo é aceito com
     probabilidade |psi(novo)|^2 / |psi(velho)|^2.
  2) Energia local: E_L(r) = -alpha^2/2 + (alpha - 1)/r
  3) A energia é a MÉDIA de E_L sobre as posições visitadas.

A IDEIA DE CUDA
  1 thread = 1 walker. Cada thread faz a sua caminhada inteira sozinha e
  grava a média de E_L dela. A CPU faz a média de todos os walkers.
"""
import sys
sys.path.insert(0, "/home/roza/Projetos/vmc-multielectron-atoms/.env/lib/python3.14/site-packages")

import os
#os.environ["NUMBA_ENABLE_CUDASIM"] = "1"    # simulador (sem GPU). Com GPU NVIDIA, apague esta linha.
os.environ["NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING"] = "1"

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, uint64, float64

np.seterr(over="ignore")   # o gerador aleatório usa o "estouro" do inteiro de propósito

# desenha os gráficos do Spyder em SVG (evita o erro KeyError: 'PNG')
try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'svg'")
except NameError:
    pass

SIMULADOR = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
U = np.uint64          # inteiro de 64 bits sem sinal (para o gerador aleatório)


# ---------------------------------------------------------------------------
# Gerador de números aleatórios: cada walker tem o SEU estado.
# uniforme() devolve um número entre 0 e 1.
# ---------------------------------------------------------------------------
@cuda.jit(device=True)
def splitmix64(z):
    z = z + U(0x9E3779B97F4A7C15)
    z = (z ^ (z >> U(30))) * U(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U(27))) * U(0x94D049BB133111EB)
    return z ^ (z >> U(31))


@cuda.jit(device=True)
def uniforme(estado):
    s = estado[0]
    s ^= s >> U(12)
    s ^= s << U(25)
    s ^= s >> U(27)
    estado[0] = s
    return float64((s * U(0x2545F4914F6CDD1D)) >> U(11)) * (1.0 / 9007199254740992.0)


# ---------------------------------------------------------------------------
# O kernel: uma thread = um walker
# ---------------------------------------------------------------------------
@cuda.jit
def vmc(alpha, passos, aquec, semente, E_walker):
    w = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x   # número deste walker
    if w >= E_walker.size:
        return

    estado = cuda.local.array(1, uint64)            # o gerador aleatório deste walker
    estado[0] = splitmix64(U(semente) + U(w))

    # posição inicial do elétron: um ponto qualquer no cubo [-1, 1]^3
    x = 2.0 * uniforme(estado) - 1.0
    y = 2.0 * uniforme(estado) - 1.0
    z = 2.0 * uniforme(estado) - 1.0
    r = math.sqrt(x * x + y * y + z * z)

    soma_EL = 0.0
    for p in range(passos):
        # 1) propõe um passo aleatório de até 1 bohr em cada direção
        xn = x + (2.0 * uniforme(estado) - 1.0)
        yn = y + (2.0 * uniforme(estado) - 1.0)
        zn = z + (2.0 * uniforme(estado) - 1.0)
        rn = math.sqrt(xn * xn + yn * yn + zn * zn)

        # 2) Metropolis: |psi(novo)|^2 / |psi(velho)|^2 = exp(-2 alpha (rn - r))
        if uniforme(estado) < math.exp(-2.0 * alpha * (rn - r)):
            x, y, z, r = xn, yn, zn, rn             # aceita: o elétron vai para lá

        # 3) depois do aquecimento, acumula a energia local
        if p >= aquec:
            soma_EL += -0.5 * alpha * alpha + (alpha - 1.0) / r

    E_walker[w] = soma_EL / (passos - aquec)        # média deste walker


# ---------------------------------------------------------------------------
# Programa principal (roda na CPU)
# ---------------------------------------------------------------------------
W = 64 if SIMULADOR else 100_000       # número de walkers (= número de threads)
PASSOS = 400 if SIMULADOR else 2000
AQUEC = PASSOS // 10                   # passos iniciais descartados
TPB = 128                              # threads por bloco
BLOCOS = (W + TPB - 1) // TPB

d_E = cuda.device_array(W)             # vetor na GPU: uma energia por walker
alphas = [0.6, 0.8, 1.0, 1.2, 1.4]
E_med, E_err = [], []

print(f"{W} walkers x {PASSOS} passos ({BLOCOS} blocos de {TPB} threads)\n")
print(" alpha    E (VMC)               exato alpha^2/2 - alpha")
for alpha in alphas:
    vmc[BLOCOS, TPB](alpha, PASSOS, AQUEC, 12345, d_E)
    E = d_E.copy_to_host()                         # traz as energias para a CPU
    media = E.mean()
    erro = E.std() / math.sqrt(W - 1)              # walkers são independentes
    E_med.append(media)
    E_err.append(erro)
    print(f" {alpha:.1f}   {media:.6f} +- {erro:.6f}    {alpha**2 / 2 - alpha:.6f}")

# gráfico
a = np.linspace(0.5, 1.5, 200)
plt.figure(figsize=(8, 5))
plt.plot(a, a**2 / 2 - a, "k", label=r"exato: $\alpha^2/2 - \alpha$")
plt.errorbar(alphas, E_med, E_err, fmt="o", label="VMC na GPU")
plt.axvline(1.0, color="gray", ls=":")
plt.xlabel(r"$\alpha$")
plt.ylabel("E (hartree)")
plt.title(r"Hidrogênio: o mínimo em $\alpha = 1$ é a energia exata, -0.5 Ha")
plt.legend()
plt.show()