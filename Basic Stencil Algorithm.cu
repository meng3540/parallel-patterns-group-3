#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// function by ahsan
void disp3DMat(float* mat, int rows, int cols, int height) {
	printf("{\n\n");
	for (int h = 0; h < height; h++) {
		printf("Slice Index % d:\n", h);
		for (int c = 0; c < cols; c++) {
			for (int r = 0; r < rows; r++) {
				int idx = h*rows*cols + c*rows + r;
				printf("%6.2f    ", mat[idx]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("}\n\n");
}

__global__ void stencil_kernel(float* in, float* out, int N, float c0, float c1, float c2, float c3, float c4, float c5, float c6) {
	unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

	//printf("i, j, k = %d, %d, %d\n", i, j, k);
	//printf("Executing Thread With Index: %d\n", i * N * N + j * N + k);

	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
		out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] + c1 * in[i * N * N + j * N + k - 1] + c2 * in[i * N * N + j * N + k + 1] + c3 * in[i * N * N + (j - 1) * N + k] + c4 * in[i * N * N + (j + 1) * N + k] + c5 * in[(i - 1) * N * N + j * N + k] + c6 * in[(i + 1) * N * N + j * N + k];
		//printf("Executing Thread With Index: %d\n", i * N * N + j * N + k);
	}
}

int main(int argc, char** argv)
{
	float* h_in; // The input (Host)
	float* h_out; // The output (Host)
	float* d_in; // The input (Device)
	float* d_out; // The output (Device)

	int N = 3; // cube size of the input

	// @@complete the size in bytes for matrix B and C; matrixA given
	int size_in = (N*N*N) * sizeof(float);
	int size_out = size_in;

	// Allocate and initialize on host
	h_in = (float*)malloc(size_in);
	h_out = (float*)malloc(size_out);

	for (int k = 0; k < N; k++) 
	{
		for (int j = 0; j < N; j++)
		{
			for (int i = 0; i < N; i++)
			{
				int idx = i * N * N + j * N + k;
				h_in[idx] = idx;
			}
		}
	}

	printf("Input Matrix:");
	disp3DMat(h_in, N, N, N);
	
	//Allocate GPU memory
	cudaMalloc((void**)&d_in, size_in);
	cudaMalloc((void**)&d_out, size_out);

	//Copy memory to the GPU here
	cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

	//@@Initialize the grid and block dimensions
	dim3 gridSize(N, N, N);
	dim3 blockSize(1, 1, 1);
	//printf("\n\nThe block dimensions are %d x %d x %d\n", gridSize.x, gridSize.y, gridSize.z);
	//printf("The grid dimensions are %d x %d x %d\n\n", blockSize.x, blockSize.y, blockSize.z);

	//@@ Launch the GPU Kernel here
	stencil_kernel << <gridSize, blockSize >> > (d_in, d_out, N, 1, 1, 1, 1, 1, 1, 1);
	cudaDeviceSynchronize();

	// error checking
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Kernel execution error: %s\n", cudaGetErrorString(err));
	}

	//Copy the GPU memory back to the CPU here
	cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

	// print result
	printf("Output Matrix:");
	disp3DMat(h_out, N, N, N);

	//Free the GPU memory here
	cudaFree(d_in);
	cudaFree(d_out);

	//free host memory
	free(h_in);
	free(h_out);

	return 0;
}