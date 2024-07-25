#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel_args.h"  // for intellisense
#include "score.h"

/**
 * Kernel to compute dot product of a single query vector relative to a batch of vectors.
 */
__global__ void multiplyAndSum(
    const float* batch,
    const float* query,
    float* dotResults,
    float* normResults,
    size_t batchSize,
    size_t vectorSize
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * vectorSize) {
        size_t batchIdx = idx / vectorSize;
        size_t vectorIdx = idx % vectorSize;
        atomicAdd(&dotResults[batchIdx], batch[idx] * query[vectorIdx]);
        atomicAdd(&normResults[batchIdx], batch[idx] * batch[idx]);
    }
}

__global__ void normalize(
    float* dotResults,
    const float* normResults,
    float queryNorm,
    size_t batchSize
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize) {
        dotResults[idx] /= (sqrtf(normResults[idx]) * queryNorm);
    }
}


void cudaCosineSimilarity(
    const float* batch,
    const float* query,
    float* results,
    size_t batchSize,
    size_t vectorSize
) {
    float queryNorm = norm(query, vectorSize);

    // allocate cuda memory
    float* cudaBatch;
    float* cudaQuery;
    float* cudaDotResults;
    float* cudaNormResults;
    size_t memSize = batchSize * sizeof(float);
    cudaMalloc(&cudaBatch, memSize * vectorSize);
    cudaMalloc(&cudaQuery, memSize);
    cudaMalloc(&cudaDotResults, memSize);
    cudaMalloc(&cudaNormResults, memSize);
    cudaMemcpy(cudaBatch, batch, memSize * vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaQuery, query, memSize, cudaMemcpyHostToDevice);
    cudaMemset(cudaDotResults, 0, memSize);
    cudaMemset(cudaNormResults, 0, memSize);

    // run kernel
    int threads = 256;
    int blocks = ((int)(batchSize * vectorSize) + threads - 1) / threads;
    multiplyAndSum KERNEL_ARGS2(blocks, threads) (cudaBatch, cudaQuery, cudaDotResults, cudaNormResults, batchSize, vectorSize);
    cudaDeviceSynchronize();

    blocks = ((int)batchSize + threads - 1) / threads;
    normalize KERNEL_ARGS2(blocks, threads) (cudaDotResults, cudaNormResults, queryNorm, batchSize);
    cudaDeviceSynchronize();

    // copy results to cpu and free cuda memory
    cudaMemcpy(results, cudaDotResults, memSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaBatch);
    cudaFree(cudaQuery);
    cudaFree(cudaDotResults);
    cudaFree(cudaNormResults);
}
