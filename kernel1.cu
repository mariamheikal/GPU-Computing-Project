#include "common.h"
#include "timer.h"
#define VERTICES_PER_BLOCK 16
#define NEIGHBORS_of_NIEGHBORS_s 6140
#define BLOCK_DIM 1024
#define THRESHOLD 80000
__global__ void jaccard_gpu1_kernel(CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d, unsigned int* numCommonNeighbors, unsigned int* neighborsOfNeighbors){
    __shared__ unsigned int numNeighborsOfNeighbors;
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
    }
    __syncthreads();
       unsigned int numVerticesPerBlock= VERTICES_PER_BLOCK*(1+(csrGraph_d->numVertices)/THRESHOLD);

       for(unsigned int v=0; v<numVerticesPerBlock; ++v){
           unsigned int vertex = numVerticesPerBlock*blockIdx.x+v;

           if(vertex< csrGraph_d->numVertices){
                for(unsigned int e=csrGraph_d->srcPtrs[vertex]; e<csrGraph_d->srcPtrs[vertex + 1]; e+=blockDim.x){

                    unsigned int edge = e+threadIdx.x;
                    if(edge<csrGraph_d->srcPtrs[vertex + 1]){

                        unsigned int neighbor = csrGraph_d->dst[edge];

                        for(unsigned int neighborEdge = csrGraph_d->srcPtrs[neighbor + 1]; neighborEdge > csrGraph_d->srcPtrs[neighbor]; --neighborEdge) {
                                unsigned int neighborOfNeighbor = csrGraph_d->dst[neighborEdge - 1];

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
                                } else {
                                    break;
                                }
                        }
                    }
                }
                __syncthreads();

                for(unsigned int i=0; i<numNeighborsOfNeighbors_s; i+=blockDim.x){

                    if(i+threadIdx.x<numNeighborsOfNeighbors_s && i+threadIdx.x<NEIGHBORS_of_NIEGHBORS_s){
                        unsigned int vertex2 = neighborsOfNeighbors_s[i+threadIdx.x];
                        if(numCommonNeighbors_s[vertex2-vertex-1] > 0) {
                            unsigned int numNeighbors = csrGraph_d->srcPtrs[vertex + 1] - csrGraph_d->srcPtrs[vertex];
                            unsigned int numNeighbors2 = csrGraph_d->srcPtrs[vertex2 + 1] - csrGraph_d->srcPtrs[vertex2];
                            float jaccardSimilarity = ((float) numCommonNeighbors_s[vertex2-vertex-1])/(numNeighbors + numNeighbors2 - numCommonNeighbors_s[vertex2-vertex-1]);
                            unsigned int j = atomicAdd(&cooMatrix_d->nnz,1);
                            cooMatrix_d->rowIdxs[j] = vertex;
                            cooMatrix_d->colIdxs[j] = vertex2;
                            cooMatrix_d->values[j] = jaccardSimilarity;
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
                            unsigned int j = atomicAdd(&cooMatrix_d->nnz,1);
                            cooMatrix_d->rowIdxs[j] = vertex;
                            cooMatrix_d->colIdxs[j] = vertex2;
                            cooMatrix_d->values[j] = jaccardSimilarity;
                            numCommonNeighbors[arrStartIndx+vertex2-NEIGHBORS_of_NIEGHBORS_s] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            if(threadIdx.x==0) {numNeighborsOfNeighbors=0; numNeighborsOfNeighbors_s=0;}
        }
    }


void jaccard_gpu1(CSRGraph* csrGraph, CSRGraph* csrGraph_d, COOMatrix* cooMatrix_d) {

    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numVertices = csrGraph->numVertices;
    unsigned int numBlocks = (numVertices+VERTICES_PER_BLOCK*(1+numVertices/THRESHOLD)-1)/(VERTICES_PER_BLOCK*(1+numVertices/THRESHOLD));
    int devID=0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devID);
    size_t sharedMemPerBlock=devProp.sharedMemPerBlock;
    printf("sharedMemPerBlock: %zu \n",sharedMemPerBlock);
    unsigned int* numCommonNeighbors;
    unsigned int* neighborsOfNeighbors;
    cudaMalloc((void**) &numCommonNeighbors, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaMalloc((void**) &neighborsOfNeighbors, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaMemset(numCommonNeighbors, 0, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaMemset(neighborsOfNeighbors, 0, numBlocks*(numVertices-NEIGHBORS_of_NIEGHBORS_s)*sizeof(unsigned int));
    cudaDeviceSynchronize();

    jaccard_gpu1_kernel<<<numBlocks, numThreadsPerBlock>>>(csrGraph_d, cooMatrix_d, numCommonNeighbors, neighborsOfNeighbors);
}


