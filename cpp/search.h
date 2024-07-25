#ifndef SEARCH_H
#define SEARCH_H

#include <vector>

std::vector<std::tuple<float, size_t>> findSimilar(
    std::vector<float>& flattenedEmbeddings,
    std::vector<float>& query,
    size_t numEmbeddings,
    size_t vectorSize,
    size_t topK,
    size_t batchSize = 65536,
    bool cuda = false
);

#endif
