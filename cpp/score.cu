#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel_args.h"  // for intellisense
#include "score.h"

/**
 * Kernel to compute dot product of a single query vector relative to a batch of vectors.
 */
__global__ void cosineSimilarityKernel(
    const float* batch,
    const float* query,
    const float* batchNorms,
    float queryNorm,
    float* results,
    size_t batchSize,
    size_t vectorSize
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * vectorSize) {
        size_t batchIdx = idx / vectorSize;
        size_t vectorIdx = idx % vectorSize;
        atomicAdd(&results[batchIdx], batch[idx] * query[vectorIdx] / batchNorms[batchIdx] / queryNorm);
    }
}

void cudaCosineSimilarity(
    const float* batch,
    const float* query,
    const float* batchNorms,
    const float queryNorm,
    float* results,
    size_t batchSize,
    size_t vectorSize
) {
    // assign cpu pointers
    float* cudaBatch;
    float* cudaQuery;
    float* cudaBatchNorms;
    float* cudaResults;
    size_t memSize = batchSize * sizeof(float);

    // allocate gpu memory and copy
    cudaMalloc(&cudaBatch, memSize * vectorSize);
    cudaMalloc(&cudaQuery, memSize);
    cudaMalloc(&cudaBatchNorms, memSize);
    cudaMalloc(&cudaResults, memSize);
    cudaMemcpy(cudaBatch, batch, memSize * vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaQuery, query, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaBatchNorms, batchNorms, memSize, cudaMemcpyHostToDevice);
    cudaMemset(cudaResults, 0, memSize);

    // run kernel
    int threads = 256;
    int blocks = ((int)(batchSize * vectorSize) + threads - 1) / threads;
    cosineSimilarityKernel KERNEL_ARGS2(blocks, threads) (cudaBatch, cudaQuery, cudaBatchNorms, queryNorm, cudaResults, batchSize, vectorSize);
    cudaDeviceSynchronize();

    // copy results to cpu and free gpu memory
    cudaMemcpy(results, cudaResults, memSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaBatch);
    cudaFree(cudaQuery);
    cudaFree(cudaResults);
}
