
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "BC.h"
#include "../graph_parser/parse.h"
//#include "util.h"

#ifdef GEM5_FUSION
#define MAX_ITERS 150
#else
#include <stdint.h>
#define MAX_ITERS INT32_MAX
#endif

void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

__global__ void
bfs_kernel(int *row, int *col, int *d, float *rho, int *cont,
	const int num_nodes, const int num_edges, const int dist)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//navigate the current layer
	if (tid < num_nodes && d[tid] == dist) {

		//get the starting and ending pointers
		//of the neighbor list

		int start = row[tid];
		int end;
		if (tid + 1 < num_nodes)
			end = row[tid + 1];
		else
			end = num_edges;

		//navigate through the neighbor list
		for (int edge = start; edge < end; edge++) {
			int w = col[edge];
			if (d[w] < 0) {
				*cont = 1;
				//traverse another layer
				d[w] = dist + 1;
			}
			//transfer the rho value to the neighbor
			if (d[w] == (dist + 1)) {
				atomicAdd(&rho[w], rho[tid]);
			}
		}
	}
}

/**
 * @brief   Back traversal
 * @param   row       CSR pointer array
 * @param   col       CSR column  array
 * @param   d         Distance array
 * @param   rho       Rho array
 * @param   sigma     Sigma array
 * @param   p         Dependency array
 * @param   cont      Termination variable
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 * @param   dist      Current traversal layer
 * @param   s         Source vertex
 * @param   bc        Betweeness Centrality array
 */

__global__ void
backtrack_kernel(int *row, int *col, int *d, float *rho, float *sigma,
	const int num_nodes, const int num_edges, const int dist,
	const int s, float* bc)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Navigate the current layer
	if (tid < num_nodes && d[tid] == dist - 1) {

		int start = row[tid];
		int end;
		if (tid + 1 < num_nodes)
			end = row[tid + 1];
		else
			end = num_edges;

		// Get the starting and ending pointers
		// of the neighbor list in the reverse graph
		for (int edge = start; edge < end; edge++) {
			int w = col[edge];
			// Update the sigma value traversing back
			if (d[w] == dist - 2)
				atomicAdd(&sigma[w], rho[w] / rho[tid] * (1 + sigma[tid]));
		}

		// Update the BC value
		if (tid != s)
			bc[tid] = bc[tid] + sigma[tid];
	}

}

/**
 * @brief   back_sum_kernel (not used)
 * @param   s         Source vertex
 * @param   dist      Current traversal layer
 * @param   d         Distance array
 * @param   sigma     Sigma array
 * @param   bc        Betweeness Centrality array
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 */
__global__ void
back_sum_kernel(const int s, const int dist, int *d, float *sigma, float *bc,
	const int num_nodes)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_nodes) {
		// If it is not the source
		if (s != tid && d[tid] == dist - 1) {
			bc[tid] = bc[tid] + sigma[tid];
		}
	}
}

/**
 * @brief   array set 1D
 * @param   s           Source vertex
 * @param   dist_array  Distance array
 * @param   sigma       Sigma array
 * @param   rho         Rho array
 * @param   num_nodes Termination variable
 */
__global__ void
clean_1d_array(const int source, int *dist_array, float *sigma, float *rho,
	const int num_nodes)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_nodes) {

		sigma[tid] = 0;

		if (tid == source) {
			// If source vertex rho = 1, dist = 0
			rho[tid] = 1;
			dist_array[tid] = 0;
		}
		else {
			// If other vertices rho = 0, dist = -1
			rho[tid] = 0;
			dist_array[tid] = -1;
		}
	}
}

/**
 * @brief   array set 2D
 * @param   p           Dependency array
 * @param   num_nodes   Number of vertices
 */
__global__ void clean_2d_array(int *p, const int num_nodes)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_nodes * num_nodes)
		p[tid] = 0;
}

/**
 * @brief   clean BC
 * @param   bc_d        Betweeness Centrality array
 * @param   num_nodes   Number of vertices
 */
__global__ void clean_bc(float *bc_d, const int num_nodes)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_nodes)
		bc_d[tid] = 0;
}

int main(int argc, char **argv)
{

	char *tmpchar;

	int num_nodes;
	int num_edges;
	bool directed = 1;

	cudaError_t err;

	if (argc == 2) {
		tmpchar = argv[1];       //graph inputfile
	}
	else {
		fprintf(stderr, "You did something wrong!\n");
		exit(1);
	}

	//tmpchar = "D:\\GraduateStudies\\ProfSinclair\\benchmarks\\pannotia\\dataset\\bc\\1k_128k.gr";

	// Parse graph and store it in a CSR format
	csr_array *csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);

	// Allocate the bc host array
	float *bc_h = (float *)malloc(num_nodes * sizeof(float));
	if (!bc_h) fprintf(stderr, "malloc failed bc_h\n");

	// Create device-side buffers
	float *bc_d, *sigma_d, *rho_d;
	int *dist_d, *stop_d;
	int *row_d, *col_d, *row_trans_d, *col_trans_d;

	// Create betweenness centrality buffers
	err = cudaMalloc(&bc_d, num_nodes * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc bc_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&dist_d, num_nodes * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc dist_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&sigma_d, num_nodes * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc sigma_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&rho_d, num_nodes * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc rho_d %s\n", cudaGetErrorString(err));
		return -1;
	}

	// Create termination variable buffer
	err = cudaMalloc(&stop_d, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc stop_d %s\n", cudaGetErrorString(err));
		return -1;
	}

	// Create graph buffers
	err = cudaMalloc(&row_d, (num_nodes + 1) * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc row_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&col_d, num_edges * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc col_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&row_trans_d, (num_nodes + 1) * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc row_trans_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&col_trans_d, num_edges * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMalloc col_trans_d %s\n", cudaGetErrorString(err));
		return -1;
	}
	//timer1 = gettime();

#ifdef GEM5_FUSION
	m5_work_begin(0, 0);
#endif

	// Copy data to device-side buffers
	err = cudaMemcpy(row_d, csr->row_array, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMemcpy row_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
		return -1;
	}

	err = cudaMemcpy(col_d, csr->col_array, num_edges * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMemcpy col_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
		return -1;
	}

	// Copy data to device-side buffers
	err = cudaMemcpy(row_trans_d, csr->row_array_t, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMemcpy row_trans_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
		return -1;
	}

	err = cudaMemcpy(col_trans_d, csr->col_array_t, num_edges * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaMemcpy col_trans_d (size:%d) => %s\n", num_nodes, cudaGetErrorString(err));
		return -1;
	}

	/*for (int i = 0; i <= num_nodes; i++) {
		row_d[i] = csr->row_array[i];
		col_d[i] = csr->col_array[i];
		row_trans_d[i] = csr->row_array_t[i];
		col_trans_d[i] = csr->col_array_t[i];
	}*/

	// Set up kernel dimensions
	int local_worksize = 128;
	dim3 threads(local_worksize, 1, 1);
	int num_blocks = (num_nodes + local_worksize - 1) / local_worksize;
	dim3 grid(num_blocks, 1, 1);

	// Initialization
	clean_bc<<< grid, threads >>>(bc_d, num_nodes);

	// Main computation loop
	for (int i = 0; i < num_nodes && i < MAX_ITERS; i++) {

		clean_1d_array<<< grid, threads >>>(i, dist_d, sigma_d, rho_d,
			num_nodes);

		// Depth of the traversal
		int dist = 0;
		// Termination variable
		int stop = 1;

		// Traverse the graph from the source node i
		do {
			stop = 0;

			// Copy the termination variable to the device
			cudaMemcpy(stop_d, &stop, sizeof(int), cudaMemcpyHostToDevice);

			bfs_kernel<<< grid, threads >>>(row_d, col_d, dist_d, rho_d, stop_d,
				num_nodes, num_edges, dist);

			// Copy back the termination variable from the device
			cudaMemcpy(&stop, stop_d, sizeof(int), cudaMemcpyDeviceToHost);

			// Another level
			dist++;

		} while (stop);

		cudaThreadSynchronize();

		// Traverse back from the deepest part of the tree
		while (dist) {
			backtrack_kernel<<< grid, threads >>>(row_trans_d, col_trans_d,
				dist_d, rho_d, sigma_d,
				num_nodes, num_edges, dist, i,
				bc_d);

			// Back one level
			dist--;
		}
		cudaThreadSynchronize();

	}
	cudaThreadSynchronize();
	//timer4 = gettime();

	// Copy back the results for the bc array
	err = cudaMemcpy(bc_h, bc_d, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "ERROR: read buffer bc_d (%s)\n", cudaGetErrorString(err));
		return -1;
	}

#ifdef GEM5_FUSION
	m5_work_end(0, 0);
#endif

	//timer2 = gettime();

	//printf("kernel + memcopy time = %lf ms\n", (timer4 - timer3) * 1000);
	//printf("kernel execution time = %lf ms\n", (timer2 - timer1) * 1000);

#if 0
	//dump the results to the file
	print_vectorf(bc_h, num_nodes);
#endif

	// Clean up the host-side buffers
	free(bc_h);
	free(csr->row_array);
	free(csr->col_array);
	free(csr->data_array);
	free(csr->row_array_t);
	free(csr->col_array_t);
	free(csr->data_array_t);
	free(csr);

	// Clean up the device-side buffers
	cudaFree(bc_d);
	cudaFree(dist_d);
	cudaFree(sigma_d);
	cudaFree(rho_d);
	cudaFree(stop_d);
	cudaFree(row_d);
	cudaFree(col_d);
	cudaFree(row_trans_d);
	cudaFree(col_trans_d);

	return 0;

}

void print_vector(int *vector, int num)
{
	for (int i = 0; i < num; i++)
		printf("%d: %d \n", i + 1, vector[i]);
	printf("\n");
}

void print_vectorf(float *vector, int num)
{

	FILE * fp = fopen("result.out", "w");
	if (!fp) {
		printf("ERROR: unable to open result.txt\n");
	}

	for (int i = 0; i < num; i++) {
		fprintf(fp, "%f\n", vector[i]);
	}

	fclose(fp);

}
