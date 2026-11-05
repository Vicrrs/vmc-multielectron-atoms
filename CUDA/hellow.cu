#include "cuda_runtime.h"
#include <stdio.h>

// kernel: roda na GPU
__global__ void meuKernel() {
    printf("Hello da GPU! thread %d\n", threadIdx.x);
}

// main: roda na CPU (host)
int main() {
    printf("Hello world da CPU\n");
    meuKernel<<<1, 1>>>();      // <<<numero de blocos, threads por bloco>>>
    cudaDeviceSynchronize();    // espera a GPU terminar
    return 0;
}
