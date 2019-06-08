#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#ifdef GEM5_FUSION
#define MAX_ITERS 128
#else
#include <stdint.h>
#define MAX_ITERS INT32_MAX
#endif

__global__ void
increment(int *memblock, int k, int modOp){
   int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
   //int loc = id%modOp;
   for (int i=0; i<k; i++){
       memblock[id] += id;
   }
}


int main(int argc, char **argv){
   int num_threads;
   int num_cus;

   if (argc >= 2){
       num_threads = atoi(argv[1]);
   } else {
       num_threads = 1024;
   }

   if (argc >= 3){
       num_cus = atoi(argv[2]);
   } else {
       num_cus = 1;
   }

   int *memBlock = (int*)malloc(sizeof(int)*num_threads);
   for (int i=0; i<num_threads; i++){
       memBlock[i] = 0;
   }

#ifdef GEM5_FUSION
   m5_work_begin(0,0);
#endif
   
   int threads_per_block = 32; //num_threads/num_cus;
   int num_blocks = num_threads/threads_per_block;

   dim3 threads_dim(threads_per_block, 1, 1);
   dim3 grid_dim(num_blocks, 1, 1);
   hipLaunchKernelGGL(HIP_KERNEL_NAME(increment), grid_dim, threads_dim, 0, 0, memBlock, 1, threads_per_block);
   
   for(int i=0; i<num_threads; i++){
       printf("%d %d\n", i, memBlock[i]);
   }

#ifdef GEM5_FUSION
   m5_work_end(0, 0);
#endif

   return 0;
}

