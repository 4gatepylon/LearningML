// cuBLAS (cublasGemmEx: bf16 in, fp32 accumulate, bf16 out): the reference and the bar to beat.
#include "harness.cuh"

MATMUL_TESTS(Cublas, matmul_cublas)

int main(int argc, char **argv) { return run_matmul("cublas", matmul_cublas, argc, argv); }
