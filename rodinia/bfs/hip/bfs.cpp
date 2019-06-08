#include "hip/hip_runtime.h"
/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 512

#ifdef GEM5_FUSION
#include <stdint.h>
extern "C" {
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#define hipMalloc(ptr, size) ({fprintf(stderr, "MALLOC\n"); int* tptr = (int*) malloc(size); if (!tptr) {fprintf(stderr, "MALLOC FAILED");}; *ptr = tptr; hipSuccess;})
#define hipMemcpy(dest, src, size, hiptype) ({fprintf(stderr, "MEMCPY\n"); if (hiptype == hipMemcpyHostToDevice) {memcpy(dest, src, size);} else {memcpy(dest, src, size);} hipSuccess;})
#define hipDeviceSynchronize() ({;})
#define hipFree(addr) ({free(addr);})

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

#include "kernel.h"
#include "kernel2.h"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

    char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d %d",&id,&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    

	printf("Read File\n");

	//Copy the Node list to device memory
	Node* d_graph_nodes;
	hipMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	hipMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, hipMemcpyHostToDevice) ;

	//Copy the Edge List to device Memory
	int* d_graph_edges;
	hipMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
	hipMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, hipMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	hipMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	hipMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, hipMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	hipMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	hipMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, hipMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	hipMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	hipMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, hipMemcpyHostToDevice) ;

	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	// allocate device memory for result
	int* d_cost;
	hipMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);

	//make a bool to check if the execution is over
	bool *d_over;
	hipMalloc( (void**) &d_over, sizeof(bool));


#ifdef GEM5_FUSION
    m5_work_begin(0, 0);
#endif

	hipMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, hipMemcpyHostToDevice) ;

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	//do
	{
		//if no thread changes this value then the loop stops
		stop=false;
		hipMemcpy( d_over, &stop, sizeof(bool), hipMemcpyHostToDevice) ;
		hipLaunchKernelGGL(HIP_KERNEL_NAME(Kernel), dim3(grid), dim3(threads), 0 , 0,  d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
		// check if kernel execution generated and error
		

		//hipLaunchKernelGGL(HIP_KERNEL_NAME(Kernel2), dim3(grid), dim3(threads), 0 , 0,  d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
	
		hipDeviceSynchronize();	
		hipMemcpy( h_updating_graph_mask, d_updating_graph_mask, sizeof(bool), hipMemcpyDeviceToHost);		

		hipMemcpy( &stop, d_over, sizeof(bool), hipMemcpyDeviceToHost) ;
		k++;
	}
	//while(stop);
	hipDeviceSynchronize();

	for (int i=0; i<no_of_nodes; i++){
	    if (h_updating_graph_mask[i] == true) 
		printf("mask[%d]: %d\n", i, h_updating_graph_mask[i]);
	}

	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	hipMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, hipMemcpyDeviceToHost) ;

#ifdef GEM5_FUSION
    m5_work_end(0, 0);
#endif

	//Store the result into a file
	// FILE *fpo = fopen("result.txt","w");
	FILE *fpo = stdout;
	//for(int i=0;i<no_of_nodes&&i<1000;i++)
		//fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	// printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	hipFree(d_graph_nodes);
	hipFree(d_graph_edges);
	hipFree(d_graph_mask);
	hipFree(d_updating_graph_mask);
	hipFree(d_graph_visited);
	hipFree(d_cost);
}
