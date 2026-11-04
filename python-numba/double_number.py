import os
os.environ["NUMA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda

@cuda.jit
def dobrar(v):
    # i = posicao da thread
    i = cuda.blockIdx.x * cuda.blockDim.x +
