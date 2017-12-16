/*
*  file name: matrix.cu
*
*  matrix.cu contains the code that realize some common used matrix operations in CUDA
*
*  this is a toy program for learning CUDA, some functions are reusable in other project
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 3

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters:
&a GPU device pointer to a m X n matrix (A)
&b GPU device pointer to a n X k matrix (B)
&c GPU device output purpose pointer to a m X k matrix (C)
to store the result

Note:
grid and block should be configured as:
dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void MatAdd(int A[][N], int B[][N], int C[][N], int m)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	printf("%d col : %d row :\n", i, j );
	printf("end \n");
	/// non margine
	int sum = 0;
	if (i != 0 && i != m - 1 && j != 0 && j != N - 1) {
			for (int j = 0; j < N; j++)
			{
				sum += A[i - 1][j] + B[i - 1][j];
				sum += A[i + 1][j] + B[i + 1][j];
			}
			C[i][j] += sum;
			sum = 0;
			for (int i = 0; i < m; i++)
			{
				sum += A[i][j + 1] + B[i][j + 1];
				sum += A[i][j - 1] + B[i][j - 1];
			}
			C[i][j] += sum;
	}
	////marginea din dreapta fara prima linie de sus
	sum = 0;
	if (i != 0 && i != m - 1 && j != 0 && j == N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i - 1][j] + B[i - 1][j];
			sum += A[i + 1][j] + B[i + 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j - 1] + B[i][j - 1];
		}
		C[i][j] += sum;
	}
	////marginea din dreapta + primul el de sus 
	sum = 0;
	if (i == 0 && i != m - 1 && j != 0 && j == N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i + 1][j] + B[i + 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j - 1] + B[i][j - 1];
		}
		C[i][j] += sum;
	}

	////marginea din dreapta + primul el de jos 
	sum = 0;
	if (i != 0 && i == m - 1 && j != 0 && j == N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i - 1][j] + B[i - 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j - 1] + B[i][j - 1];
		}
		C[i][j] += sum;
	}

	////marginea din stanga doar primul element de sus 
	sum = 0;
	if (i == 0 && i != m - 1 && j == 0 && j != N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i + 1][j] + B[i + 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j + 1] + B[i][j + 1];
		}
		C[i][j] += sum;
	}
	////marginea din stanga doar primul element de jos 
	sum = 0;
	if (i != 0 && i == m - 1 && j == 0 && j != N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i - 1][j] + B[i - 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j + 1] + B[i][j + 1];
		}
		C[i][j] += sum;
	}
	
	////marginea din stanga fara elementul de sus si jos
	sum = 0;
	if (i != 0 && i != m - 1 && j == 0 && j != N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i - 1][j] + B[i - 1][j];
			sum += A[i + 1][j] + B[i + 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j + 1] + B[i][j + 1];
		}
		C[i][j] += sum;
	}

	////marginea de sus 
	sum = 0;
	if (i == 0 && i != m - 1 && j != 0 && j != N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i + 1][j] + B[i + 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j + 1] + B[i][j + 1];
			sum += A[i][j - 1] + B[i][j - 1];
		}
		C[i][j] += sum;
	}

	////marginea de jos 
	sum = 0;
	if (i != 0 && i == m - 1 && j != 0 && j != N - 1) {
		for (int j = 0; j < N; j++)
		{
			sum += A[i - 1][j] + B[i - 1][j];
		}
		C[i][j] += sum;
		sum = 0;
		for (int i = 0; i < m; i++)
		{
			sum += A[i][j + 1] + B[i][j + 1];
			sum += A[i][j - 1] + B[i][j - 1];
		}
		C[i][j] += sum;
	}

	
}



/*
*********************************************************************
function name: main

description: test and compare

parameters:
none

return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{

	/* Fixed seed for illustration */

	const int m = 6;
	float gpu_elapsed_time_ms;

	// some events to count the execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start to count execution time of GPU version
	cudaEventRecord(start, 0);
	// Allocate memory space on the device 
	int A[m][N];
	int B[m][N];
	int C[m][N];


	for (int i = 0; i<m; i++) {
		for (int j = 0; j<N; j++) {
			A[i][j] = 1;
		}
		printf("\n");
	}
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<N; j++) {
			B[i][j] = 1;
		}
		printf("\n");
	}

	for (int i = 0; i<m; i++) {
		for (int j = 0; j<N; j++) {
			C[i][j] = 0;
		}
		printf("\n");
	}


	int(*pA)[N], (*pB)[N], (*pC)[N];

	cudaMalloc((void**)&pA, (m*N) * sizeof(int));
	cudaMalloc((void**)&pB, (m*N) * sizeof(int));
	cudaMalloc((void**)&pC, (m*N) * sizeof(int));

	cudaMemcpy(pA, A, (m*N) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, B, (m*N) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, C, (m*N) * sizeof(int), cudaMemcpyHostToDevice);


	// Launch kernel 

	int numBlocks = 1;
	dim3 threadsPerBlock(m, N);
	MatAdd << <numBlocks, threadsPerBlock >> >(pA,pB,pC,m);

	// Transefr results from device to host 
	cudaMemcpy(C, pC, (m*N) * sizeof(int), cudaMemcpyDeviceToHost);


	cudaThreadSynchronize();
	// time counting terminate
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	printf("Time elapsed on matrix calcul of %f matrix", gpu_elapsed_time_ms);
	printf("\n");
	printf("A = \n");
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<N; j++) {
			printf("%d ", A[i][j]);
		}
		printf("\n");
	}
	printf("B = \n");
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<N; j++) {
			printf("%d ", B[i][j]);
		}
		printf("\n");
	}

	int i, j; 
	
	printf("C = \n");
	for (i = 0; i<m; i++) {
		for (j = 0; j<N; j++) {
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}

	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);

	printf("\n");

	system("PAUSE");
	return 0;
}
