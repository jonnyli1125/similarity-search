#include "heap.h"

void heapAdd(ScoreIndexHeap& heap, ScoreIndexPair& item, size_t maxSize) {
    if (heap.size() < maxSize) {
        heap.push(item);
    } else if (item > heap.top()) {
        heap.pop();
        heap.push(item);
    }
}

std::vector<ScoreIndexPair> heapTopK(ScoreIndexHeap& heap, size_t k) {
    std::vector<ScoreIndexPair> result;
    ScoreIndexHeap tempHeap = heap;
    while (!tempHeap.empty()) {
        result.push_back(tempHeap.top());
        tempHeap.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}
