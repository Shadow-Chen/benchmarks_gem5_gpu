#include <stdio.h>
#include <string.h>
//#include <cuda.h>
#include <hip/hip_runtime.h>
/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 64

#define hipMalloc(ptr,size) host_malloc(ptr,size)
#define hipMemcpy(dest,src,size,hipcpytype) host_memcpy(dest,src,size,hipcpytype)
#define hipDeviceSynchronize() ({;})
#define hipFree(ptr) host_free(ptr);

void host_malloc(void **ptr, size_t size){
    fprintf(stderr, "Malloc\n");
    int *tptr = (int*)malloc(size);
    if (!tptr) fprintf(stderr, "malloc failed\n");
    *ptr = tptr; hipSuccess;
}

void host_memcpy(void *dest, void *src, size_t size, hipMemcpyKind hipcpytype){
    if (hipcpytype == hipMemcpyHostToDevice)
        memcpy(dest, src, size);
    else
        memcpy(src, dest, size);
}

void host_free(void *ptr){
    free(ptr);
}


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
__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
{
	int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters:
			&a GPU device pointer to a n X n matrix (A)
			&b GPU device pointer to a n X n matrix (B)
			&c GPU device output purpose pointer to a n X n matrix (C)
			to store the result
Note:
	grid and block should be configured as:

		dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
		dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n)
{
	__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

	int row = hipBlockIdx_y * BLOCK_SIZE + hipThreadIdx_y;
	int col = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
	int tmp = 0;
	int idx;

	for (int sub = 0; sub < hipGridDim_x; ++sub)
	{
		idx = row * n + sub * BLOCK_SIZE + hipThreadIdx_x;
		if (idx >= n * n)
		{
			// n may not divisible by BLOCK_SIZE
			tile_a[hipThreadIdx_y][hipThreadIdx_x] = 0;
		}
		else
		{
			tile_a[hipThreadIdx_y][hipThreadIdx_x] = d_a[idx];
		}

		idx = (sub * BLOCK_SIZE + hipThreadIdx_y) * n + col;
		if (idx >= n * n)
		{
			tile_b[hipThreadIdx_y][hipThreadIdx_x] = 0;
		}
		else
		{
			tile_b[hipThreadIdx_y][hipThreadIdx_x] = d_b[idx];
		}
		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			tmp += tile_a[hipThreadIdx_y][k] * tile_b[k][hipThreadIdx_x];
		}
		__syncthreads();
	}
	if (row < n && col < n)
	{
		d_result[row * n + col] = tmp;
	}
}

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters:
			&mat_in GPU device pointer to a rows X cols matrix
			&mat_out GPU device output purpose pointer to a cols X rows matrix
			to store the result
Note:
	grid and block should be configured as:
		dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
		dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
	unsigned int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	unsigned int idy = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

	if (idx < cols && idy < rows)
	{
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	}
}
/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU,
			 for validating GPU results

parameters:
			&a CPU host pointer to a m X n matrix (A)
			&b CPU host pointer to a n X k matrix (B)
			&c CPU host output purpose pointer to a m X k matrix (C)
			to store the result
return: none
*********************************************************************
*/
void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			int tmp = 0.0;
			for (int h = 0; h < n; ++h)
			{
				tmp += h_a[i * n + h] * h_b[h * k + j];
			}
			h_result[i * k + j] = tmp;
		}
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
	int m, n, k;
	/* Fixed seed for illustration */
	srand(3333);
	//printf("please type in m n and k\n");
	//scanf("%d %d %d", &m, &n, &k);

	m = atoi(argv[1]);
	n = atoi(argv[2]);
	k = atoi(argv[3]);

	// allocate memory in host RAM, h_cc is used to store CPU result
	int *h_a, *h_b, *h_c, *h_cc;
	h_a = (int*)malloc(sizeof(int)*m*n);
	h_b = (int*)malloc(sizeof(int)*n*k);
	h_c = (int*)malloc(sizeof(int)*m*k);
	h_cc = (int*)malloc(sizeof(int)*m*k);

	// random initialize matrix A
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			h_a[i * n + j] = rand() % 1024;
		}
	}

	// random initialize matrix B
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			h_b[i * k + j] = rand() % 1024;
		}
	}

	int *d_a, *d_b, *d_c;
	//hipMalloc((void **)&d_a, sizeof(int)*m*n);
	//hipMalloc((void **)&d_b, sizeof(int)*n*k);
	//hipMalloc((void **)&d_c, sizeof(int)*m*k);

	d_a = (int*)malloc(sizeof(int)*m*n);
	d_b = (int*)malloc(sizeof(int)*n*k);
	d_c = (int*)malloc(sizeof(int)*m*k);

	// copy matrix A and B from host to device memory
	//hipMemcpy(d_a, h_a, sizeof(int)*m*n, hipMemcpyHostToDevice);
	//hipMemcpy(d_b, h_b, sizeof(int)*n*k, hipMemcpyHostToDevice);

	memcpy(d_a, h_a, sizeof(int)*m*n);
	memcpy(d_b, h_b, sizeof(int)*n*k);

	dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	printf("Grid size %d %d block size %d", (k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);

	hipLaunchKernelGGL(HIP_KERNEL_NAME(gpu_matrix_mult), dim3(dimGrid), dim3(dimBlock), 0, 0,
					d_a, d_b, d_c, m, n, k);
	// Transefr results from device to host 
	//hipMemcpy(h_c, d_c, sizeof(int)*m*k, hipMemcpyDeviceToHost);
	memcpy(h_c, d_c, sizeof(int)*m*k);

	//hipDeviceSynchronize();
	// start the CPU version
	/*cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

	// validate results computed by GPU
	int all_ok = 1;
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
			if (h_cc[i*k + j] != h_c[i*k + j])
			{
				all_ok = 0;
			}
		}
		printf("\n");
	}

	// roughly compute speedup
	if (all_ok)
	{
		printf("all results are correct!!!\n");
	}
	else
	{
		printf("incorrect results\n");
	}*/

	// free memory
	/*hipFree(d_a);
	hipFree(d_b);
	hipFree(d_c);
	hipHostFree(h_a);
	hipHostFree(h_b);
	hipHostFree(h_c);
	hipHostFree(h_cc);*/
	return 0;
}
