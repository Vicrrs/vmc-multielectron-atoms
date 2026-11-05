"""
Processando uma Matriz 2D (Grades bidimensionais)

Quando trabalhamos com imagens, simulações físicas 2D ou matrizes, é mais fácil 
organizar as threads em 2D com (x, y) em vez de apenas um índice linear.
"""

import numpy as np
from numba import cuda

@cuda.jit
def eleva_matriz_ao_quadrado(matriz):
    # cuda.grid(2) retorna uma dupla (x, y) com a posição 2D da thread
    x, y =  cuda.grid(2)

    # Verifica limites em 2D (linhas e colunas)
    linhas, colunas = matriz.shape
    if x < linhas and y < colunas:
        matriz[x, y] = matriz[x, y] ** 2

# MAtriz 4x4 no Host (CPU)
h_matriz = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float32)

# Envia para GPU
d_matriz = cuda.to_device(h_matriz)

# Para 2D, passamos tuplas para definir blocos e grid:
# Bloco de 2x2 threads (4 threads por bloco)
threads_por_bloco = (2, 2)

# Grid de 2x2 blocos (4 blocos total) -> total: 4x4 = 16 threads
blocos_por_grid = (2, 2)

# Executa o kernel 2D
eleva_matriz_ao_quadrado[blocos_por_grid, threads_por_bloco](d_matriz)

# Traz de volta e exibe
resultado = d_matriz.copy_to_host()
print("Matriz ao quadrado:\n", resultado)
