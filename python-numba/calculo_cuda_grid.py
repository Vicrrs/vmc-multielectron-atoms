"""
Entendendo o cálculo do cuda.grid(1) manualmente

O atalho cuda.grid(1) faz um cálculo simples por baixo dos panos usando o ID do bloco e da thread. 
Entender a fórmula ajuda a visualizar como a GPU se organiza:

posicao_global = (blockIdx.x * blockDim.x) + threadIdx.x
"""

import numpy as np
from numba import cuda

@cuda.jit
def exibe_geometria(saida):
    # Atributos nativos do CUDA:
    # cuda.threadIdx.x -> ID da thread DENTRO do bloco atual (ex: 0 a 3)
    # cuda.blockIdx.x  -> ID do bloco atual (ex: 0 ou 1)
    # cuda.blockDim.x  -> Número de threads por bloco (neste caso, 4)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    # Calculo manual do índice global:
    pos_manual = (bid * bdim) + tid

    # Atalho do Numba (faz exatamente a mesma coisa):
    pos_atalho = cuda.grid(1)

    if pos_atalho < saida.size:
        saida[pos_atalho] = pos_manual  # Armazena o resultado no array de saída

saida_gpu = cuda.device_array(8, dtype=np.int32)  # Array de saída na GPU

# Lançamos 2 blocos, cada um com 4 threads = 8 threads no total
exibe_geometria[2, 4](saida_gpu)

resultado = saida_gpu.copy_to_host()  # Copia o resultado de volta para a CPU
print("Resultado do cálculo manual do índice global:", resultado)
