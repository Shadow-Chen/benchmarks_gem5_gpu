#include "hip/hip_runtime.h"
/************************************************************************************\ 
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
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
#include "../graph_parser/parse.h"
#include "kernel_max.h"
#include "kernelpf.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define RANGE 2048
#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

void print_vector(int *vector, int num);

void
initPathfinder(int rows, int cols, int pyramid_height)
{
	data = new int[rows*cols];
	wall = new int*[rows];
	
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    	{
        	for (int j = 0; j < cols; j++)
        	{	
            		wall[i][j] = rand() % 10;
        	}
    	}
}

int main(int argc, char **argv)
{
    char *tmpchar;

    int num_nodes;
    int num_edges;
    int file_format = 1;
    bool directed = 0;

    // inputs for color and pathfinder
    if (argc == 5) {
        tmpchar = argv[1];  //graph inputfile
        cols = atoi(argv[2]);
	rows = atoi(argv[3]);
        pyramid_height=atoi(argv[4]);
    } else {
        fprintf(stderr, "Usage: inp_file_name row_len col_len pyramid_height\n");
        exit(1);
    }

    /* Setting Up Color */
    srand(7);
    // Allocate the CSR structure
    csr_array *csr;

    // Parse graph file and store into a CSR format
    if (file_format == 1)
        csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
    else if (file_format == 0)
        csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);
    else {
        printf("reserve for future");
        exit(1);
    }

    // Allocate the vertex value array
    int *node_value = (int *)malloc(num_nodes * sizeof(int));
    if (!node_value) fprintf(stderr, "node_value malloc failed\n");
    // Allocate the color array
    int *color = (int *)malloc(num_nodes * sizeof(int));
    if (!color) fprintf(stderr, "color malloc failed\n");

    // Initialize all the colors to -1
    // Randomize the value for each vertex
    for (int i = 0; i < num_nodes; i++) {
        color[i] = -1;
        node_value[i] = rand() % RANGE;
    }

    int *row_d;
    int *col_d;
    int *max_d;

    int *color_d;
    int *node_value_d;
    int *stop_d;

    row_d = (int*)malloc(sizeof(int)*num_nodes);
    col_d = (int*)malloc(sizeof(int)*num_edges);
    stop_d = (int*)malloc(sizeof(int));
    color_d = (int*)malloc(sizeof(int)*num_nodes);
    node_value_d = (int*)malloc(sizeof(int)*num_nodes);
    max_d = (int*)malloc(sizeof(int)*num_nodes);

    memcpy(color_d, color, sizeof(int)*num_nodes);
    memcpy(max_d, color, sizeof(int)*num_nodes);
    memcpy(row_d, csr->row_array, sizeof(int)*num_nodes);
    memcpy(col_d, csr->col_array, sizeof(int)*num_edges);
    memcpy(node_value_d, node_value, sizeof(int)*num_nodes);

    /* Setting up Pathfinder */
    initPathfinder(rows,cols,pyramid_height);

    // pyramid parameters 
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    //hipMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    //hipMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    //hipMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
    
    //hipMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, hipMemcpyDeviceToHost);
#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

    // Set up kernel dimensions
    // color
    int block_size = 64;
    int num_blocks = (num_nodes + block_size - 1) / block_size;
    dim3 threads(block_size,  1, 1);
    dim3 grid(num_blocks, 1,  1);

    // pathfinder
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    int stop = 1;
    int graph_color = 1;
    int t = 0;
    int src = 1, dst = 0;

    printf("Launching Kernels\n");
    while (stop || t < rows-1) {
        /* Launch Pathfinder */
        int temp = src;
        src = dst;
        dst = temp;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(dynproc_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, 
                MIN(pyramid_height, rows-t-1), gpuWall, gpuResult[src], gpuResult[dst], cols,rows, t, borderCols);
        t += pyramid_height;

        /* Launch color */
        stop = 0;
	    memcpy(stop_d, &stop, sizeof(int)); 
        // Launch the color kernel 1
        hipLaunchKernelGGL(HIP_KERNEL_NAME(color1), dim3(grid), dim3(threads ), 0, 0, row_d, col_d, node_value_d, color_d,
                                     stop_d, max_d, graph_color, num_nodes,
                                     num_edges);
    	//hipDeviceSynchronize();
        // Launch the color kernel 2
        hipLaunchKernelGGL(HIP_KERNEL_NAME(color2), dim3(grid), dim3(threads ), 0, 0, node_value_d, color_d, max_d, graph_color,
                                     num_nodes, num_edges);

	    memcpy(&stop, stop_d, sizeof(int));
        graph_color++;

    }
    
    //hipDeviceSynchronize();
    memcpy(color, color_d, sizeof(int)*num_nodes);
    //memcpy(result, gpuResult[final_ret], sizeof(int)*cols);

    printf("graph color %d destination %d\n",graph_color,dst);

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

#if 1
    // Dump the color array into an output file
    print_vector(color, num_nodes);
#endif

    // Free host-side buffers
    //free(node_value);
    //free(color);
    //csr->freeArrays();
    //free(csr);

    // Free CUDA buffers
    /*hipFree(row_d);
    hipFree(col_d);
    hipFree(max_d);
    hipFree(color_d);
    hipFree(node_value_d);
    hipFree(stop_d);*/

    return 0;
}

void print_vector(int *vector, int num)
{
    FILE * fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++)
        //printf("%d: %d\n", i + 1, vector[i]);
	fprintf(fp, "%d: %d\n", i + 1, vector[i]);

    fclose(fp);
}
