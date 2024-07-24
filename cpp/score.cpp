#include "score.h"

void cpuCosineSimilarity(
    const float* batch,
    const float* query,
    const float* batchNorms,
    const float queryNorm,
    float* results,
    size_t batchSize,
    size_t vectorSize
) {
    for (size_t i = 0; i < batchSize; i++) {
        results[i] = 0;
        for (size_t j = 0; j < vectorSize; j++) {
            results[i] += batch[i * vectorSize + j] * query[j];
        }
        results[i] /= (batchNorms[i] * queryNorm);
    }
}

float norm(std::vector<float> vec) {
    float norm = 0.0f;
    for (float x : vec) {
        norm += x * x;
    }
    return (float)sqrt(norm);
}
