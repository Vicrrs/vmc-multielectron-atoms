# Somar dois vetores elemento por elemento usando Numba para acelerar
# Vamos somar dois arrays A e B e guardar em C

import numpy as np
from numba import cuda
import math

# @cuda.jit indica que esta funcao sera compilada para rodar na GPU
@cuda.jit
def soma_vetores(a, b, c):
    # cuda.grid(1) pega a posicao global da thread atual no array (0, 1, 2, ..., n-1)
    pos = cuda.grid(1)

    # Verificação de segurança: como lançamos mais threads que o tamanho do array,
    # garantimos que a thread fora do limite do array nao tentem acessar memória inválida.
    if pos < c.size:
        c[pos] = a[pos] + b[pos]

# Execucao no Host (CPU)

N = 1000

# 1. Cria os dados no Host (CPU)
h_a = np.ones(N, dtype=np.float32)      # Array de 1 s
h_b = np.full(N, 5.0, dtype=np.float32) # Array de 5 s
h_c = np.zeros(N, dtype=np.float32)    # Array de 0 s para receber o resultado

# 2. Copia os dados do Host para o Device (GPU)
d_a = cuda.to_device(h_a)
d_b = cuda.to_device(h_b)
d_c = cuda.to_device(h_c)

# 3. Configura a geometria da GPU
# O padrão comum na NVIDIA é usar 128, 256 ou 512 threads por bloco.
threads_por_bloco = 256
# Calculamos quantos blocos precisamos para cobrir o tamanho de N (sempre arredondando para cima)
blocos_por_grid = math.ceil(N / threads_por_bloco)

# 4. Executa o Kernel na GPU
soma_vetores[blocos_por_grid, threads_por_bloco](d_a, d_b, d_c)

# 5. Copia o resultado de volta para o Host (CPU)
h_c = d_c.copy_to_host()

print("Primeiros 5 resultados: ", h_c[:5])
