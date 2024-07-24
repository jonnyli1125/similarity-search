#ifndef HEAP_H
#define HEAP_H

#include <vector>
#include <queue>

typedef std::tuple<float, size_t> ScoreIndexPair; // (similarity score, index)
typedef std::priority_queue<
    std::tuple<float, size_t>,
    std::vector<std::tuple<float, size_t>>,
    std::greater<std::tuple<float, size_t>>
> ScoreIndexHeap; // min-heap to keep top k elements

void heapAdd(ScoreIndexHeap& heap, ScoreIndexPair& item, size_t maxSize);

std::vector<ScoreIndexPair> heapTopK(ScoreIndexHeap& heap, size_t k);

#endif
