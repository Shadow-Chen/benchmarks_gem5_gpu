//#include <hip/hip_runtime.h>
//#include "device_launch_parameters.h"

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C"{
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

int addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = hipThreadIdx_x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    //hipLaunchKernelGGL(addKernel, dim3(1), dim3(arraySize), 0, 0, c, a, b);
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int cudaStatus;

    dev_c = (int*)malloc(sizeof(int)*arraySize);
    dev_a = (int*)malloc(sizeof(int)*arraySize);
    dev_b = (int*)malloc(sizeof(int)*arraySize);

    // Copy input vectors from host memory to GPU buffers.
    for (int i = 0; i < arraySize; i++) {
	dev_a[i] = a[i];
	dev_b[i] = b[i];
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(addKernel), dim3(1), dim3(arraySize), 0, 0, dev_c, dev_a, dev_b);

    for (int i = 0; i < arraySize; i++) {
	c[i] = dev_c[i];
    }

    // Add vectors in parallel.
    /*int cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }*/

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
int addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int cudaStatus;

    dev_c = (int*)malloc(sizeof(int)*size);
    dev_a = (int*)malloc(sizeof(int)*size);
    dev_b = (int*)malloc(sizeof(int)*size);

    // Copy input vectors from host memory to GPU buffers.
    for (int i = 0; i < size; i++) {
	dev_a[i] = a[i];
	dev_b[i] = b[i];
    }

    hipLaunchKernelGGL(HIP_KERNEL_NAME(addKernel), dim3(1), dim3(size), 0, 0, dev_c, dev_a, dev_b);

    // hipDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    //cudaStatus = hipDeviceSynchronize();
    //if (cudaStatus != hipSuccess) {
    //    fprintf(stderr, "hipDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //    goto Error;
    //}

    // Copy output vector from GPU buffer to host memory.
    for (int i = 0; i < size; i++) {
	c[i] = dev_c[i];
    }

    /*cudaStatus = hipMemcpy(c, dev_c, size * sizeof(int), hipMemcpyDeviceToHost);
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy failed!");
        goto Error;
    }*/

//Error:
    //hipFree(dev_c);
    //hipFree(dev_a);
    //hipFree(dev_b);
    
    return cudaStatus;
}

