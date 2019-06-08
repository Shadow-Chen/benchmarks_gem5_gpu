#include "hip/hip_runtime.h"
#include "stdlib.h"
#include "stdio.h"


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


//#define hipMalloc(ptr, size) ({fprintf(stderr, "MALLOC\n"); int* tptr = (int*) malloc(size); if (!tptr) {fprintf(stderr, "MALLOC FAILED");}; *ptr = tptr; hipSuccess;})
//#define hipMemcpy(dest, src, size, hiptype) ({fprintf(stderr, "MEMCPY\n"); if (hiptype == hipMemcpyHostToDevice) {memcpy(dest, src, size);} else {memcpy(dest, src, size);} hipSuccess;})
//#define hipDeviceSynchronize() ({;})
//#define hipFree(addr) ({free(addr);})

