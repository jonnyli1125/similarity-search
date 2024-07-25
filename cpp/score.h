#ifndef SCORE_H
#define SCORE_H

#include <vector>

void cudaCosineSimilarity(
    const float* batch,
    const float* query,
    float* results,
    size_t batchSize,
    size_t vectorSize
);
void cpuCosineSimilarity(
    const float* batch,
    const float* query,
    float* results,
    size_t batchSize,
    size_t vectorSize
);
float norm(const float* vec, size_t len);

#endif
