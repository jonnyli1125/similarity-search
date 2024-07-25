#include "score.h"

void cpuCosineSimilarity(
    const float* batch,
    const float* query,
    float* results,
    size_t batchSize,
    size_t vectorSize
) {
    float queryNorm = norm(query, vectorSize);
    std::vector<float> batchNorms(batchSize, 0.0f);
    for (size_t i = 0; i < batchSize; i++) {
        results[i] = 0;
        for (size_t j = 0; j < vectorSize; j++) {
            float val = batch[i * vectorSize + j];
            results[i] += val * query[j];
            batchNorms[i] += val * val;
        }
        results[i] /= (sqrtf(batchNorms[i]) * queryNorm);
    }
}

float norm(const float* vec, size_t len) {
    float norm = 0.0f;
    for (size_t i = 0; i < len; i++) {
        norm += vec[i] * vec[i];
    }
    return sqrtf(norm);
}
