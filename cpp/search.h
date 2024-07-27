#ifndef SEARCH_H
#define SEARCH_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

std::vector<std::tuple<float, size_t>> findSimilar(
    const float* flattenedEmbeddings,
    const float* query,
    size_t numEmbeddings,
    size_t vectorSize,
    size_t topK,
    size_t batchSize = 65536,
    bool cuda = true
);

std::vector<std::tuple<float, size_t>> findSimilarNumpy(
    pybind11::array_t<float>& flattenedEmbeddings,
    pybind11::array_t<float>& query,
    size_t topK,
    size_t batchSize = 65536,
    bool cuda = true
);

#endif
