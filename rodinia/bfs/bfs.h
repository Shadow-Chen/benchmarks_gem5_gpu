#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512

struct Node
{
	int starting;
	int no_of_edges;
};


__global__ void
Kernel( Node *, int *, bool *, bool *, bool *, int *, int, int*);

__global__ void
Kernel2( bool *, bool *, bool *, bool *, int, int*);
