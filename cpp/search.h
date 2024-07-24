#ifndef SEARCH_H
#define SEARCH_H

#include <vector>

std::vector<std::tuple<float, size_t>> findSimilar(
    std::vector<float>& flattenedEmbeddings,
    std::vector<float>& norms,
    std::vector<float>& query,
    size_t numRows,
    size_t vectorSize,
    size_t topK,
    size_t batchSize = 4096,
    bool cuda = false
);

#endif
