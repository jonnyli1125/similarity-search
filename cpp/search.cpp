#include <stdexcept>

#include "search.h"
#include "score.h"
#include "heap.h"

using namespace std;
namespace py = pybind11;

/**
 * Return score and index of top k most similar vectors in embeddings relative to the query vector.
 */
vector<ScoreIndexPair> findSimilar(
    const float* flattenedEmbeddings,
    const float* query,
    size_t numEmbeddings,
    size_t vectorSize,
    size_t topK,
    size_t batchSize,
    bool useCuda
) {
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
        if (useCuda) {
            cudaCosineSimilarity(
                flattenedEmbeddings + i * vectorSize,
                query,
                scores.data(),
                currentBatchSize,
                vectorSize
            );
        } else {
            cpuCosineSimilarity(
                flattenedEmbeddings + i * vectorSize,
                query,
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

vector<ScoreIndexPair> findSimilarNumpy(
    py::array_t<float>& embeddings,
    py::array_t<float>& query,
    size_t topK,
    size_t batchSize,
    bool useCuda
) {
    py::buffer_info embeddingsBuf = embeddings.request();
    if (embeddingsBuf.ndim != 2) {
        throw invalid_argument("embeddings must be a 2D array");
    }
    size_t numEmbeddings = embeddingsBuf.shape[0];
    size_t vectorSize = embeddingsBuf.shape[1];
    py::buffer_info queryBuf = query.request();
    if (queryBuf.ndim != 1) {
        throw invalid_argument("query must be a 1D array");
    }
    if (queryBuf.shape[0] != vectorSize) {
        throw invalid_argument("query dim must be same as embeddings vector dim");
    }
    const float* embeddingsPtr = static_cast<float *>(embeddingsBuf.ptr);
    const float* queryPtr = static_cast<float *>(queryBuf.ptr);
    return findSimilar(embeddingsPtr, queryPtr, numEmbeddings, vectorSize, topK, batchSize, useCuda);
}

using namespace pybind11::literals;
PYBIND11_MODULE(similarity_search, m) {
    m.def(
        "find_similar",
        &findSimilarNumpy,
        "Return score and index of top k most similar vectors in embeddings relative to the query vector.",
        "embeddings"_a,
        "query"_a,
        "top_k"_a,
        "batch_size"_a=65536,
        "use_cuda"_a=true
    );
}
