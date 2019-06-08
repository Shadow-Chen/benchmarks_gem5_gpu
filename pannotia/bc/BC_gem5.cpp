#include "hip/hip_runtime.h"
/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include "BC.h"
#include "../graph_parser/util.h"
//#include "kernel.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#ifdef GEM5_FUSION
#define MAX_ITERS 150
#else
#include <stdint.h>
#define MAX_ITERS INT32_MAX
#endif

void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

/**
 * @brief   Breadth-first traversal
 * @param   row       CSR pointer array
 * @param   col       CSR column  array
 * @param   d         Distance array
 * @param   rho       Rho array
 * @param   p         Dependency array
 * @param   cont      Termination variable
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 * @param   dist      Current traversal layer
 */

__global__ void
bfs_kernel(int *row, int *col, int *d, float *rho, int *cont,
           const int num_nodes, const int num_edges, const int dist)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

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
            //if (d[w] == (dist + 1)) {
                //atomicAdd(&rho[w], rho[tid]);
            //}
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

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
            //if (d[w] == dist - 2)
                //atomicAdd(&sigma[w], rho[w] / rho[tid] * (1 + sigma[tid]));
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid < num_nodes) {

        sigma[tid] = 0;

        if (tid == source) {
            // If source vertex rho = 1, dist = 0
            rho[tid] = 1;
            dist_array[tid] = 0;
        } else {
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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

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
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid < num_nodes)
        bc_d[tid] = 0;
}

int main(int argc, char **argv)
{

    char *tmpchar;

    int num_nodes;
    int num_edges;
    bool directed = 1;
    hipError_t err;

    if (argc == 2) {
        tmpchar     = argv[1];       //graph inputfile
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    // Parse graph and store it in a CSR format
    csr_array *csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);

    // Allocate the bc host array
    float *bc_h = (float *)malloc(num_nodes * sizeof(float));
    if (!bc_h) fprintf(stderr, "malloc failed bc_h\n");

    // Create device-side buffers
    float *sigma_h, *rho_h;
    int *dist_h, *stop;

    dist_h = (int*)malloc(sizeof(int)*num_nodes);
    sigma_h = (float*)malloc(sizeof(float)*num_nodes);
    rho_h = (float*)malloc(sizeof(float)*num_nodes);
    stop = (int*)malloc(sizeof(int)); 

    double timer1, timer2;
    double timer3, timer4;

    //timer1 = gettime();

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif
    //timer3 = gettime();

    // Set up kernel dimensions
    int local_worksize = 128;
    dim3 threads(local_worksize, 1, 1);
    int num_blocks = (num_nodes + local_worksize - 1) / local_worksize;
    dim3 grid(num_blocks, 1, 1);

    // Initialization
    hipLaunchKernelGGL(HIP_KERNEL_NAME(clean_bc), dim3(grid), dim3(threads ), 0, 0, bc_h, num_nodes);

    // Main computation loop
    for (int i = 0; i < num_nodes && i < MAX_ITERS; i++) {

        hipLaunchKernelGGL(HIP_KERNEL_NAME(clean_1d_array), dim3(grid), dim3(threads ), 0, 0, i, dist_h, sigma_h, rho_h,
                                            num_nodes);

        // Depth of the traversal
        int dist = 0;
        // Termination variable
        //int stop = 1;

        // Traverse the graph from the source node i
        do {
            *stop = 0;

            // Copy the termination variable to the device
            hipLaunchKernelGGL(HIP_KERNEL_NAME(bfs_kernel), dim3(grid), dim3(threads ), 0, 0, csr->row_array, csr->col_array, dist_h, rho_h, stop,
                                            num_nodes, num_edges, dist);
            printf("stop %d\n",*stop);
            // Another level
            dist++;
        } while (*stop) ;

        hipDeviceSynchronize();

        // Traverse back from the deepest part of the tree
        while (dist) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(backtrack_kernel), dim3(grid), dim3(threads ), 0, 0, csr->row_array_t, csr->col_array_t,
                                                dist_h, rho_h, sigma_h,
                                                num_nodes, num_edges, dist, i,
                                                bc_h);

            // Back one level
            dist--;
        }
        hipDeviceSynchronize();

    }
    hipDeviceSynchronize();
    //timer4 = gettime();

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

    //timer2 = gettime();

    printf("kernel + memcopy time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel execution time = %lf ms\n", (timer2 - timer1) * 1000);

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

    free(dist_h);
    free(sigma_h);
    free(rho_h);
    free(stop);

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

