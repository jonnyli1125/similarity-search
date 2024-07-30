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
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batchSize * vectorSize; idx += blockDim.x * gridDim.x) {
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
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batchSize; idx += blockDim.x * gridDim.x) {
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
    size_t batchMemSize = batchSize * sizeof(float);
    size_t queryMemSize = vectorSize * sizeof(float);
    size_t totalMemSize = batchSize * vectorSize * sizeof(float);
    cudaMalloc(&cudaBatch, totalMemSize);
    cudaMalloc(&cudaQuery, queryMemSize);
    cudaMalloc(&cudaDotResults, batchMemSize);
    cudaMalloc(&cudaNormResults, batchMemSize);
    cudaMemcpy(cudaBatch, batch, totalMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaQuery, query, queryMemSize, cudaMemcpyHostToDevice);
    cudaMemset(cudaDotResults, 0, batchMemSize);
    cudaMemset(cudaNormResults, 0, batchMemSize);

    // run kernel
    int threads = 256;
    int blocks = std::min(65535, ((int)(batchSize * vectorSize) + threads - 1) / threads);
    multiplyAndSum KERNEL_ARGS2(blocks, threads) (cudaBatch, cudaQuery, cudaDotResults, cudaNormResults, batchSize, vectorSize);
    cudaDeviceSynchronize();

    blocks = std::min(65535, ((int)batchSize + threads - 1) / threads);
    normalize KERNEL_ARGS2(blocks, threads) (cudaDotResults, cudaNormResults, queryNorm, batchSize);
    cudaDeviceSynchronize();

    // copy results to cpu and free cuda memory
    cudaMemcpy(results, cudaDotResults, batchMemSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaBatch);
    cudaFree(cudaQuery);
    cudaFree(cudaDotResults);
    cudaFree(cudaNormResults);
}
