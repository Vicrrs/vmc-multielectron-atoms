import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

from numba import cuda

@cuda.jit
def hellow_thread():
    t = cuda.threadIdx.x    # posicao da thread DENTRO do bloco
    b = cuda.blockIdx.x     # numero do bloco
    bd = cuda.blockDim.x    # quantas threads tem em cada bloco
    i = b * bd + t          # indice global unico dessa thread

    print("bloco", b, "| thread local", t, "| indice global", i)

# lanca 2 blocos de 4 threads = 8 threads no total
hellow_thread[2, 4]()

