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
//#include <sys/time.h>
#include "../graph_parser/parse.h"
//#include "../graph_parser/util.h"
#include "kernel_max.h"

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define RANGE 2048

void print_vector(int *vector, int num);

int main(int argc, char **argv)
{
    char *tmpchar;

    int num_nodes;
    int num_edges;
    int file_format = 1;
    bool directed = 0;

    if (argc == 3) {
        tmpchar = argv[1];  //graph inputfile
        file_format = atoi(argv[2]); //graph format
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

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

    int *max = (int*)malloc(num_nodes*sizeof(int));

    // Initialize all the colors to -1
    // Randomize the value for each vertex
    for (int i = 0; i < num_nodes; i++) {
        color[i] = -1;
        node_value[i] = rand() % RANGE;

	//
	max[i] = -1;
    }

    //int *row_d = (int*)malloc(sizeof(int)*num_nodes);
    //int *col_d = (int*)malloc(sizeof(int)*num_edges);
    //int *max_d = (int*)malloc(sizeof(int)*num_nodes);

    //int *color_d = (int*)malloc(sizeof(int)*num_nodes);
    //int *node_value_d = (int*)malloc(sizeof(int)*num_nodes);
    //int *stop_d = (int*)malloc(sizeof(int));

#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif
   
    //memcpy(color_d, color, sizeof(int)*num_nodes);
    //memcpy(max_d, color, sizeof(int)*num_nodes);
    //memcpy(row_d, csr->row_array, sizeof(int)*num_nodes);
    //memcpy(col_d, csr->col_array, sizeof(int)*num_edges);
    //memcpy(node_value_d, node_value, sizeof(int)*num_nodes);

    int block_size = 64;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    // Set up kernel dimensions
    dim3 threads(block_size,  1, 1);
    dim3 grid(num_blocks, 1,  1);

    int *stop = (int*)malloc(sizeof(int));
    *stop = 1;
    int graph_color = 1;

    // Main computation loop
    while (*stop) {

        *stop = 0;

        // Copy the termination variable to the device
        //memcpy(stop_d, &stop, sizeof(int));
	
        // Launch the color kernel 1
        hipLaunchKernelGGL(HIP_KERNEL_NAME(color1), dim3(grid), dim3(threads ), 0, 0, csr->row_array, csr->col_array, node_value, color, stop, max, graph_color, num_nodes, num_edges);

        // Launch the color kernel 2
        hipLaunchKernelGGL(HIP_KERNEL_NAME(color2), dim3(grid), dim3(threads ), 0, 0, node_value, color, max, graph_color, num_nodes, num_edges);

	//memcpy(&stop, stop_d, sizeof(int));
        // Increment the color for the next iter
        graph_color++;

    }
    //hipDeviceSynchronize();
    //memcpy(color, color_d, num_nodes * sizeof(int));
 
#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

   // Print out color and timing statistics
    printf("total number of colors used: %d\n", graph_color);

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
        fprintf(fp, "%d: %d\n", i + 1, vector[i]);

    fclose(fp);
}
