import numpy as np
from numba import cuda

@cuda.jit
def dobra(v):
    i = cuda.grid(1)
    if i < v.size:
        v[i] = 2 * v[i]

v = np.array([1.9, 2.0, 3.0, 4.0, 5.0])

# 1. Envia os dados para a GPU
v_device = cuda.to_device(v)

# 2. Executa o kernel na GPU
dobra[2, 4](v_device)

# 3. Copia o resultado de volta para o Host (CPU)
v_resultado = v_device.copy_to_host()

print(v_resultado)
# Saída esperada: [ 3.8  4.   6.   8.  10. ]