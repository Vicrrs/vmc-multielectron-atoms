import os
# Corrigido: NUMBA com 'B'
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda

@cuda.jit
def dobra(v):
    i = cuda.grid(1)
    if i < v.size:
        v[i] = 2 * v[i]

v = np.array([1.9, 2.0, 3.0, 4.0, 5.0])
dobra[2, 4](v)  # Nota: faltava passar o argumento '(v)' no kernel call também!
print(v)
# Saída esperada: [ 3.8  4.   6.   8.  10. ]