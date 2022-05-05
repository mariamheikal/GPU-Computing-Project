#include "common.h"
#include "timer.h"
#define NEIGHBORS_of_NIEGHBORS_s 6138
#define BLOCK_DIM 1024
#define WARP_SIZE 64

__global__ void jaccard_gpu3_kernel(CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d, unsigned int* numCommonNeighbors, unsigned int* neighborsOfNeighbors, unsigned int* vertexPointer){
    __shared__ unsigned int numNeighborsOfNeighbors;
    __shared__ unsigned int j;
    __shared__ unsigned int vertex;
    __shared__ unsigned int numNeighborsOfNeighbors_s;
    __shared__ unsigned int numCommonNeighbors_s[NEIGHBORS_of_NIEGHBORS_s];
    __shared__ unsigned int neighborsOfNeighbors_s[NEIGHBORS_of_NIEGHBORS_s];
    
    unsigned int arrStartIndx = blockIdx.x*(csrGraph_d->numVertices-NEIGHBORS_of_NIEGHBORS_s);

    for(unsigned int i=0; i<NEIGHBORS_of_NIEGHBORS_s; i+=blockDim.x){
        numCommonNeighbors_s[i+threadIdx.x]=0;
        neighborsOfNeighbors_s[i+threadIdx.x]=0;
    }
    if(threadIdx.x==0) {
        numNeighborsOfNeighbors=0;
        numNeighborsOfNeighbors_s=0;
        vertex = atomicAdd(&vertexPointer[0],1);
    }
    __syncthreads();

    while(vertex<csrGraph_d->numVertices){
        for(unsigned int e=csrGraph_d->srcPtrs[vertex]; e<csrGraph_d->srcPtrs[vertex + 1]; e+=blockDim.x/WARP_SIZE){
            
            unsigned int edge = e+threadIdx.x/WARP_SIZE;

            if(edge<csrGraph_d->srcPtrs[vertex + 1]){
                
                unsigned int neighbor = csrGraph_d->dst[edge];
                
                for(long neighborEdge = csrGraph_d->srcPtrs[neighbor + 1]; neighborEdge > (long) csrGraph_d->srcPtrs[neighbor]; neighborEdge-=WARP_SIZE) {
                    if (neighborEdge - (threadIdx.x % WARP_SIZE) - 1 > csrGraph_d->srcPtrs[neighbor]){
                        
                        unsigned int neighborOfNeighbor = csrGraph_d->dst[neighborEdge - 1 -threadIdx.x%WARP_SIZE];
                        if(neighborOfNeighbor > vertex) {
                            
                            if(neighborOfNeighbor<vertex+NEIGHBORS_of_NIEGHBORS_s+1){
                                unsigned int oldVal = atomicAdd(&numCommonNeighbors_s[neighborOfNeighbor-vertex-1],1);
                                if(oldVal == 0) {
                                    neighborsOfNeighbors_s[atomicAdd(&numNeighborsOfNeighbors_s,1)] = neighborOfNeighbor;
                                }
                            }
                            else{
                                unsigned int oldVal = atomicAdd(&numCommonNeighbors[arrStartIndx+neighborOfNeighbor-NEIGHBORS_of_NIEGHBORS_s],1);
                                if(oldVal == 0) {
                                    neighborsOfNeighbors[arrStartIndx+atomicAdd(&numNeighborsOfNeighbors,1)] = neighborOfNeighbor;
                                }
                            }
                                        
                                        
                        } else break;
                                        
                    }
                }
            }

        }
        __syncthreads();
               
        if (threadIdx.x==0) j = atomicAdd(&cooMatrix_d->nnz,numNeighborsOfNeighbors_s+numNeighborsOfNeighbors);
                
        __syncthreads();

        for(unsigned int i=0; i<numNeighborsOfNeighbors_s; i+=blockDim.x){

            if(i+threadIdx.x<numNeighborsOfNeighbors_s && i+threadIdx.x<NEIGHBORS_of_NIEGHBORS_s){
                unsigned int vertex2 = neighborsOfNeighbors_s[i+threadIdx.x];
                if(numCommonNeighbors_s[vertex2-vertex-1] > 0) {
                    unsigned int numNeighbors = csrGraph_d->srcPtrs[vertex + 1] - csrGraph_d->srcPtrs[vertex];
                    unsigned int numNeighbors2 = csrGraph_d->srcPtrs[vertex2 + 1] - csrGraph_d->srcPtrs[vertex2];
                    float jaccardSimilarity = ((float) numCommonNeighbors_s[vertex2-vertex-1])/(numNeighbors + numNeighbors2 - numCommonNeighbors_s[vertex2-vertex-1]);
                    cooMatrix_d->rowIdxs[j+i+threadIdx.x] = vertex;
                    cooMatrix_d->colIdxs[j+i+threadIdx.x] = vertex2;
                    cooMatrix_d->values[j+i+threadIdx.x] = jaccardSimilarity;
                    numCommonNeighbors_s[vertex2-vertex-1] = 0;
                }
            }
        }

        for(unsigned int i=0; i<numNeighborsOfNeighbors; i+=blockDim.x){
            if(i+threadIdx.x<numNeighborsOfNeighbors){
                unsigned int vertex2 = neighborsOfNeighbors[arrStartIndx+i+threadIdx.x];
                if(numCommonNeighbors[arrStartIndx+vertex2-NEIGHBORS_of_NIEGHBORS_s] > 0) {
                    unsigned int numNeighbors = csrGraph_d->srcPtrs[vertex + 1] - csrGraph_d->srcPtrs[vertex];
                    unsigned int numNeighbors2 = csrGraph_d->srcPtrs[vertex2 + 1] - csrGraph_d->srcPtrs[vertex2];
                    float jaccardSimilarity = ((float) numCommonNeighbors[arrStartIndx+vertex2-NEIGHBORS_of_NIEGHBORS_s])/(numNeighbors + numNeighbors2 - numCommonNeighbors[arrStartIndx+vertex2-NEIGHBORS_of_NIEGHBORS_s]);
                    cooMatrix_d->rowIdxs[j+numNeighborsOfNeighbors_s+i+threadIdx.x] = vertex;
                    cooMatrix_d->colIdxs[j+numNeighborsOfNeighbors_s+i+threadIdx.x] = vertex2;
                    cooMatrix_d->values[j+numNeighborsOfNeighbors_s+i+threadIdx.x] = jaccardSimilarity;
                    numCommonNeighbors[arrStartIndx+vertex2-NEIGHBORS_of_NIEGHBORS_s] = 0;
                }
            }
        }

        __syncthreads();
        if(threadIdx.x==0) {
            numNeighborsOfNeighbors=0; 
            numNeighborsOfNeighbors_s=0; 
            vertex = atomicAdd(&vertexPointer[0],1);
        }
        __syncthreads();
    }
}

void jaccard_gpu3(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d) {


    int devID=0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devID);
    
    int multiProcessorCount=devProp.multiProcessorCount;
    int maxThreadsPerMultiProcessor=devProp.maxThreadsPerMultiProcessor;
    unsigned int numBlocks = ((multiProcessorCount*maxThreadsPerMultiProcessor)+BLOCK_DIM-1)/BLOCK_DIM;
    unsigned int numVertices=csrGraph->numVertices;
    
    unsigned int* numCommonNeighbors;
    unsigned int* neighborsOfNeighbors;
    unsigned int* vertexPointer;
    
    cudaMalloc((void**) &numCommonNeighbors, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaMalloc((void**) &neighborsOfNeighbors, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaMalloc((void**) &vertexPointer, sizeof(unsigned int));

    cudaMemset(vertexPointer, 0, sizeof(unsigned int));
    cudaMemset(numCommonNeighbors, 0, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaMemset(neighborsOfNeighbors, 0, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));

    cudaDeviceSynchronize();

    jaccard_gpu3_kernel<<<numBlocks, BLOCK_DIM>>>(csrGraph_d, cooMatrix_d, numCommonNeighbors, neighborsOfNeighbors, vertexPointer);
    cudaDeviceSynchronize();
    cudaFree(numCommonNeighbors);
    cudaFree(neighborsOfNeighbors);
}
