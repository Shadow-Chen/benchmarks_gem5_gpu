/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>

// Declaration, forward
void runTest(int argc, char **argv);

__global__ void
testKernel(int *g_odata)
{
	// access thread id
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Test various atomic instructions

	// Arithmetic atomic instructions

	// Atomic addition
	atomicAdd(&g_odata[0], 10);

	// Atomic subtraction (final should be 0)
	atomicSub(&g_odata[1], 10);

	// Atomic exchange
	atomicExch(&g_odata[2], tid);

	// Atomic maximum
	atomicMax(&g_odata[3], tid);

	// Atomic minimum
	atomicMin(&g_odata[4], tid);

	// Atomic increment (modulo 17+1)
	atomicInc((unsigned int *)&g_odata[5], 17);

	// Atomic decrement
	atomicDec((unsigned int *)&g_odata[6], 137);

	// Atomic compare-and-swap
	atomicCAS(&g_odata[7], tid - 1, tid);

	// Bitwise atomic instructions

	// Atomic AND
	atomicAnd(&g_odata[8], 2 * tid + 7);

	// Atomic OR
	atomicOr(&g_odata[9], 1 << tid);

	// Atomic XOR
	atomicXor(&g_odata[10], tid);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);
    exit(0);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    unsigned int numThreads = 256;
    unsigned int numBlocks = 64;
    unsigned int numData = 11;
    unsigned int memSize = sizeof(int) * numData;

    //allocate mem for the result on host side
    int *hOData = (int *) malloc(memSize);

    //initialize the memory
    for (unsigned int i = 0; i < numData; i++)
        hOData[i] = 0;

    //To make the AND and XOR tests generate something other than 0...
    hOData[8] = hOData[10] = 0xff;

    // execute the kernel
    testKernel<<<numBlocks, numThreads>>>(hOData);
    getLastCudaError("Kernel execution failed");

    // Cleanup memory
    free(hOData);
    checkCudaErrors(cudaFree(dOData));
}