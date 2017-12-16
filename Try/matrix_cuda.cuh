#ifndef MATRIX_CUDA_
#define MATRIX_CUDA_

void gpu_matrix_mult(float *A, float *B, float *C, int N);
void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n);

#endif