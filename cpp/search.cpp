#include <vector>
#include <queue>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "search.h"
#include "score.h"
#include "heap.h"

#include <iostream> //debug

using namespace std;

/**
 * Return score and index of top k most similar vectors in embeddings relative to the query vector.
 */
vector<ScoreIndexPair> findSimilar(
    vector<float>& flattenedEmbeddings,
    vector<float>& query,
    size_t numEmbeddings,
    size_t vectorSize,
    size_t topK,
    size_t batchSize,
    bool cuda
) {
    // validation
    if (vectorSize != query.size()) {
        throw invalid_argument("embeddings vector size must be equal to query vector size");
    }
    if (batchSize == 0) {
        batchSize = numEmbeddings;
    }

    // min heap to keep track of top k vectors
    ScoreIndexHeap heap;

    // loop over chunks in dataset
    for (size_t i = 0; i < numEmbeddings; i += batchSize) {
        size_t currentBatchSize = min(batchSize, numEmbeddings - i);

        // get cosine similarity on this batch
        vector<float> scores(currentBatchSize);
        if (cuda) {
            cudaCosineSimilarity(
                flattenedEmbeddings.data() + i * vectorSize,
                query.data(),
                scores.data(),
                currentBatchSize,
                vectorSize
            );
        } else {
            cpuCosineSimilarity(
                flattenedEmbeddings.data() + i * vectorSize,
                query.data(),
                scores.data(),
                currentBatchSize,
                vectorSize
            );
        }

        // update heap to keep track of top k
        for (size_t j = 0; j < currentBatchSize; j++) {
            heapAdd(heap, ScoreIndexPair(scores[j], i + j), topK);
        }
    }
    return heapTopK(heap, topK);
}

using namespace pybind11::literals;
PYBIND11_MODULE(similarity_search, m) {
    m.def(
        "find_similar",
        &findSimilar,
        "Return score and index of top k most similar vectors in embeddings relative to the query vector.",
        "flattened_embeddings"_a,
        "query"_a,
        "num_embeddings"_a,
        "vector_size"_a,
        "top_k"_a,
        "batch_size"_a=65536,
        "cuda"_a=false
    );
}
