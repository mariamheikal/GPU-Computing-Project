
#include "common.h"
#include "timer.h"
#define VERTICES_PER_BLOCK 32

__global__ void jaccard_gpu0_kernel(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d, unsigned int* numCommonNeighbors, unsigned int* neighborsOfNeighbors){
    __shared__ unsigned int numNeighborsOfNeighbors;
    unsigned int arrStartIndx = blockIdx.x*csrGraph_d->numVertices;
    
    if(threadIdx.x==0) {
        numNeighborsOfNeighbors=0;
    }
    __syncthreads();
     
       for(unsigned int v=0; v<VERTICES_PER_BLOCK; ++v){ 
           unsigned int vertex = VERTICES_PER_BLOCK*blockIdx.x+v;
            
           if(vertex< csrGraph_d->numVertices){
                for(unsigned int e=csrGraph_d->srcPtrs[vertex]; e<csrGraph_d->srcPtrs[vertex + 1]; e+=blockDim.x){            
        
                    unsigned int edge = e+threadIdx.x;
                    if(edge<csrGraph_d->srcPtrs[vertex + 1]){

                        unsigned int neighbor = csrGraph_d->dst[edge];
                   
                        for(unsigned int neighborEdge = csrGraph_d->srcPtrs[neighbor + 1]; neighborEdge > csrGraph_d->srcPtrs[neighbor]; --neighborEdge) {
                                unsigned int neighborOfNeighbor = csrGraph_d->dst[neighborEdge - 1];
                                if(neighborOfNeighbor > vertex) {
                                        unsigned int oldVal = atomicAdd(&numCommonNeighbors[arrStartIndx+neighborOfNeighbor],1);  
                                        if(oldVal == 0) {
                                            neighborsOfNeighbors[arrStartIndx+atomicAdd(&numNeighborsOfNeighbors,1)] = neighborOfNeighbor;
                                        }
                                } else {
                                    break;
                                }
                        } 
                    }
                }
                __syncthreads();

                for(unsigned int i=0; i<numNeighborsOfNeighbors; i+=blockDim.x){

                    if(i+threadIdx.x<numNeighborsOfNeighbors){
                        unsigned int vertex2 = neighborsOfNeighbors[arrStartIndx+i+threadIdx.x];
                        if(numCommonNeighbors[arrStartIndx+vertex2] > 0) {
                            unsigned int numNeighbors = csrGraph_d->srcPtrs[vertex + 1] - csrGraph_d->srcPtrs[vertex];
                            unsigned int numNeighbors2 = csrGraph_d->srcPtrs[vertex2 + 1] - csrGraph_d->srcPtrs[vertex2];
                            float jaccardSimilarity = ((float) numCommonNeighbors[arrStartIndx+vertex2])/(numNeighbors + numNeighbors2 - numCommonNeighbors[arrStartIndx+vertex2]);
                            unsigned int j = atomicAdd(&cooMatrix_d->nnz,1);
                            cooMatrix_d->rowIdxs[j] = vertex;
                            cooMatrix_d->colIdxs[j] = vertex2;
                            cooMatrix_d->values[j] = jaccardSimilarity;
                            numCommonNeighbors[arrStartIndx+vertex2] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            if(threadIdx.x==0) numNeighborsOfNeighbors=0;
        }
    }


void jaccard_gpu0(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d) {
    
    unsigned int avgNumEdges =  csrGraph->numEdges/csrGraph->numVertices;
    unsigned int numThreadsPerBlock = avgNumEdges>128?avgNumEdges:128;
    unsigned int numVertices = csrGraph->numVertices;
    unsigned int numBlocks = (numVertices+VERTICES_PER_BLOCK-1)/VERTICES_PER_BLOCK;

    unsigned int* numCommonNeighbors; 
    unsigned int* neighborsOfNeighbors; 
    cudaMalloc((void**) &numCommonNeighbors, numBlocks*numVertices*sizeof(unsigned int));
    cudaMalloc((void**) &neighborsOfNeighbors, numBlocks*numVertices*sizeof(unsigned int));
    cudaMemset(numCommonNeighbors, 0, numBlocks*numVertices*sizeof(unsigned int));
    cudaMemset(neighborsOfNeighbors, 0, numBlocks*numVertices*sizeof(unsigned int));
    cudaDeviceSynchronize();

    jaccard_gpu0_kernel<<<numBlocks, numThreadsPerBlock>>>(csrGraph, csrGraph_d, cooMatrix_d, numCommonNeighbors, neighborsOfNeighbors);
}



