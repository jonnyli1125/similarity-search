#include <vector>
#include <queue>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "search.h"
#include "score.h"
#include "heap.h"

using namespace std;

/**
 * Return score and index of top k most similar vectors in embeddings relative to the query vector.
 */
vector<ScoreIndexPair> findSimilar(
    vector<float>& flattenedEmbeddings,
    vector<float>& norms,
    vector<float>& query,
    size_t numRows,
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
        batchSize = numRows;
    }

    // min heap to keep track of top k vectors
    ScoreIndexHeap heap;

    // get norm of query
    float queryNorm = norm(query);

    // loop over chunks in dataset
    for (size_t i = 0; i < numRows; i += batchSize) {
        size_t currentBatchSize = min(batchSize, numRows - i);

        // get cosine similarity on this batch
        vector<float> scores(currentBatchSize);
        auto cosineSimilarity = cuda ? cudaCosineSimilarity : cpuCosineSimilarity;
        cosineSimilarity(
            flattenedEmbeddings.data() + i * vectorSize,
            query.data(),
            norms.data() + i,
            queryNorm,
            scores.data(),
            currentBatchSize,
            vectorSize
        );

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
        "norms"_a,
        "query"_a,
        "num_rows"_a,
        "vector_size"_a,
        "top_k"_a,
        "batch_size"_a=4096,
        "cuda"_a=false
    );
}
