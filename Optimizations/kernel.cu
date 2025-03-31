#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// optimization mode 
// 0 -- no optimization
// 1 -- simple tiling
// 2 -- thread coarsening
// 3 -- register tiling
#define OPTIMIZATION_MODE 0

#define CUBEDIM 512
#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

// block dimension for regular kernel
#define BLOCK_DIM 8

// tile dimensions for tiling kernel
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// tile dimensions for tiling and thread coarsening kernel
#define IN_TILE_DIMC 32
#define OUT_TILE_DIMC (IN_TILE_DIMC - 2)

// tile dimensions for tiling and thread coarsening kernel and register tiling
#define IN_TILE_DIMCRT IN_TILE_DIMC
#define OUT_TILE_DIMCRT (IN_TILE_DIMCRT - 2)


// matrix display function by ahsan
void disp3DMat(float* mat, int rows, int cols, int height) {
	printf("{\n\n");
	for (int h = 0; h < height; h++) {
		printf("Slice Index % d:\n", h);
		for (int c = 0; c < cols; c++) {
			for (int r = 0; r < rows; r++) {
				int idx = h*rows*cols + c*rows + r;
				printf("%3.2f    ", mat[idx]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("}\n\n");
}

// regular stencil kernel, no optimizations
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
	unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
		out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] + c1 * in[i * N * N + j * N + k - 1] + c2 * in[i * N * N + j * N + k + 1] + c3 * in[i * N * N + (j - 1) * N + k] + c4 * in[i * N * N + (j + 1) * N + k] + c5 * in[(i - 1) * N * N + j * N + k] + c6 * in[(i + 1) * N * N + j * N + k];
	}
}

// stencil kernel with tiling
__global__ void stencil_kernelT(float* in, float* out, unsigned int N) {
	int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
	int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
	int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

	__shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

	// Load data into shared memory
	if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
		in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
	}
	__syncthreads();

	// Apply the stencil operation
	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
		if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 &&threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
			out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
				c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
				c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
				c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
				c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
				c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
				c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
		}
	}
}

// stencil kernel with thread coarsening
__global__ void stencil_kernelTC(float* in, float* out, unsigned int N) {
	int iStart = blockIdx.z * OUT_TILE_DIMC;
	int j = blockIdx.y * OUT_TILE_DIMC + threadIdx.y - 1;
	int k = blockIdx.x * OUT_TILE_DIMC + threadIdx.x - 1;

	__shared__ float inPrev_s[IN_TILE_DIMC][IN_TILE_DIMC];
	__shared__ float inCurr_s[IN_TILE_DIMC][IN_TILE_DIMC];
	__shared__ float inNext_s[IN_TILE_DIMC][IN_TILE_DIMC];

	if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
		inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
	}
	if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
		inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
	}
	for (int i = iStart; i < iStart + OUT_TILE_DIMC; ++i) {
		if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
			inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
		}
		__syncthreads();

		if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
			if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIMC - 1
				&& threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIMC - 1) {
				out[i * N * N + j * N + k] = c0 * inCurr_s[threadIdx.y][threadIdx.x]
					+ c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
					+ c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
					+ c3 * inCurr_s[threadIdx.y + 1][threadIdx.x]
					+ c4 * inCurr_s[threadIdx.y - 1][threadIdx.x]
					+ c5 * inPrev_s[threadIdx.y][threadIdx.x]
					+ c6 * inNext_s[threadIdx.y][threadIdx.x];
			}
		}
		__syncthreads();
		inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
		inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
	}
}

//stencil kernel with thread coarsening and register tiling
__global__ void stencil_kernelTCRT(float* in, float* out, unsigned int N) {
	int iStart = blockIdx.z * OUT_TILE_DIMCRT;
	int j = blockIdx.y * OUT_TILE_DIMCRT + threadIdx.y - 1;
	int k = blockIdx.x * OUT_TILE_DIMCRT + threadIdx.x - 1;
	float inPrev;
	__shared__ float inCurr_s[IN_TILE_DIMCRT][IN_TILE_DIMCRT];
	float inCurr;
	float inNext;
	if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
		inPrev = in[(iStart - 1) * N * N + j * N + k];
	}
	if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
		inCurr = in[iStart * N * N + j * N + k];
		inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
	}
	for (int i = iStart; i < iStart + OUT_TILE_DIMCRT; ++i) {
		if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
			inNext = in[(i + 1) * N * N + j * N + k];
		}
		__syncthreads();
		if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
			if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIMCRT - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIMCRT - 1) {
				out[i * N * N + j * N + k] = c0 * inCurr
					+ c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
					+ c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
					+ c3 * inCurr_s[threadIdx.y + 1][threadIdx.x]
					+ c4 * inCurr_s[threadIdx.y - 1][threadIdx.x]
					+ c5 * inPrev
					+ c6 * inNext;
			}
		}
		__syncthreads();
		inPrev = inCurr;
		inCurr = inNext;
		inCurr_s[threadIdx.y][threadIdx.x] = inNext;
	}
}

int main(int argc, char** argv)
{
	// for calculating effective bandwidth
	int bytes_transferred = 0;

	float* h_in; // The input (Host)
	float* h_out; // The output (Host)
	float* d_in; // The input (Device)
	float* d_out; // The output (Device)

	// @@complete the size in bytes for matrix B and C; matrixA given
	int size_in = (CUBEDIM*CUBEDIM*CUBEDIM) * sizeof(float);
	int size_out = size_in;

	// Allocate and initialize on host
	h_in = (float*)malloc(size_in);
	h_out = (float*)malloc(size_out);

	for (int k = 0; k < CUBEDIM; k++) 
	{
		for (int j = 0; j < CUBEDIM; j++)
		{
			for (int i = 0; i < CUBEDIM; i++)
			{
				int idx = i * CUBEDIM * CUBEDIM + j * CUBEDIM + k;
				h_in[idx] = 1;
			}
		}
	}
	
	//Allocate GPU memory
	cudaMalloc((void**)&d_in, size_in);
	cudaMalloc((void**)&d_out, size_out);

	//Copy memory to the GPU here
	cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice);

	//@@Initialize the grid and block dimensions
	// grid and block dimensions for simple kernel
	dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((BLOCK_DIM + CUBEDIM - 1) / BLOCK_DIM, (BLOCK_DIM + CUBEDIM - 1) / BLOCK_DIM, (BLOCK_DIM + CUBEDIM - 1) / BLOCK_DIM);
	// grid and block dimensions for tiled kernel
	dim3 blockSizeT(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
	dim3 gridSizeT((OUT_TILE_DIM + CUBEDIM - 1) / OUT_TILE_DIM, (OUT_TILE_DIM + CUBEDIM - 1) / OUT_TILE_DIM, (OUT_TILE_DIM + CUBEDIM - 1) / OUT_TILE_DIM);
	// grid and block dimensions for thread coarsened kernel
	dim3 blockSizeTC(IN_TILE_DIMC, IN_TILE_DIMC, 1);
	dim3 gridSizeTC((OUT_TILE_DIMC + CUBEDIM - 1) / OUT_TILE_DIMC, (OUT_TILE_DIMC + CUBEDIM - 1) / OUT_TILE_DIMC, (OUT_TILE_DIMC + CUBEDIM - 1) / OUT_TILE_DIMC);
	// grid and block dimensions for thread coarsened kernel and register tiling
	dim3 blockSizeTCRT(IN_TILE_DIMCRT, IN_TILE_DIMCRT, 1);
	dim3 gridSizeTCRT((OUT_TILE_DIMCRT + CUBEDIM - 1) / OUT_TILE_DIMCRT, (OUT_TILE_DIMCRT + CUBEDIM - 1) / OUT_TILE_DIMCRT, (OUT_TILE_DIMCRT + CUBEDIM - 1) / OUT_TILE_DIMCRT);
	
	//@@ Launch the GPU Kernel here
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (OPTIMIZATION_MODE == 0) { // no optimization
		printf("\n\nRunning regular kernel...\n\n");
		cudaEventRecord(start);
		stencil_kernel << <gridSize, blockSize >> > (d_in, d_out, CUBEDIM);
		cudaEventRecord(stop);
		bytes_transferred = (CUBEDIM^3) * 8 * 4;
	}
	else if (OPTIMIZATION_MODE == 1) { // with tiling
		printf("\n\nRunning tiled kernel...\n\n");
		cudaEventRecord(start);
		stencil_kernelT << <gridSizeT, blockSizeT >> > (d_in, d_out, CUBEDIM);
		cudaEventRecord(stop);
	}
	else if (OPTIMIZATION_MODE == 2) { // with thread coarsening
		printf("\n\nRunning tiled and thread coarsened kernel...\n\n");
		cudaEventRecord(start);
		stencil_kernelTC << <gridSizeTC, blockSizeTC >> > (d_in, d_out, CUBEDIM);
		cudaEventRecord(stop);
	}
	else if (OPTIMIZATION_MODE == 3) { // with register tiling
		printf("\n\nRunning register tiled and thread coarsened kernel...\n\n");
		cudaEventRecord(start);
		stencil_kernelTCRT << <gridSizeTCRT, blockSizeTCRT >> > (d_in, d_out, CUBEDIM);
		cudaEventRecord(stop);
	}
	else {
		printf("\n\nIncorrect optimization mode, will run regular kernel\n\n");
		cudaEventRecord(start);
		stencil_kernel << <gridSize, blockSize >> > (d_in, d_out, CUBEDIM);
		cudaEventRecord(stop);
	}

	//Copy the GPU memory back to the CPU here
	cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// error checking
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Kernel execution error: %s\n", cudaGetErrorString(err));
	}

	// print results if size is less than or equal to 16
	if (CUBEDIM <= 16) {
		printf("Input Matrix:\n");
		disp3DMat(h_in, CUBEDIM, CUBEDIM, CUBEDIM);
		printf("Output Matrix:\n");
		disp3DMat(h_out, CUBEDIM, CUBEDIM, CUBEDIM);
	}

	//Free the GPU memory here
	cudaFree(d_in);
	cudaFree(d_out);

	//free host memory
	free(h_in);
	free(h_out);

	// print kernel execution time
	printf("\n\nKernel execution time: %.4f milliseconds\n\n", milliseconds);
	// print effecitve bandwidth
	printf("\n\nEffective Bandwidth (GB/s): %f\n\n", bytes_transferred / milliseconds / 1e6);
	// print block and grid dims
	printf("\n\nThe block dimensions are %d x %d x %d\n", blockSize.x, blockSize.y, blockSize.z);
	printf("The grid dimensions are %d x %d x %d\n\n", gridSize.x, gridSize.y, gridSize.z);
	printf("\n\nThe block dimensions with tiling are %d x %d x %d\n", blockSizeT.x, blockSizeT.y, blockSizeT.z);
	printf("The grid dimensions with tiling are %d x %d x %d\n\n", gridSizeT.x, gridSizeT.y, gridSizeT.z);
	printf("\n\nThe block dimensions with tiling and thread coarsening are %d x %d x %d\n", blockSizeTC.x, blockSizeTC.y, blockSizeTC.z);
	printf("The grid dimensions with tiling and thread coarsening are %d x %d x %d\n\n", gridSizeTC.x, gridSizeTC.y, gridSizeTC.z);

	return 0;
}