#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "search.h"

using namespace std;
using namespace std::chrono;

/**
 * Runs similarity search on a dummy dataset of 1 million random vectors and prints out the time elapsed in seconds.
 */
int main(int argc, char* argv[]) {
    const size_t numRows = 1<<20;
    const size_t vectorSize = 1<<10;
    const size_t topK = 10;
    bool showResults = argc > 1 && (strcmp(argv[1], "-r") == 0 || strcmp(argv[1], "--results") == 0);

    // create random vectors
    cout << "Creating " << numRows << " random vectors of size " << vectorSize << endl;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    vector<float> flattenedEmbeddings(numRows * vectorSize);
    vector<float> norms(numRows);
    for (size_t i = 0; i < numRows; i++) {
        float norm = 0.0f;
        for (size_t j = 0; j < vectorSize; j++) {
            float value = dis(gen);
            flattenedEmbeddings[i * vectorSize + j] = value;
            norm += value * value;
        }
        norms[i] = sqrt(norm);
    }

    // create dummy query
    cout << "Creating dummy query vector" << endl;
    vector<float> query(vectorSize);
    for (size_t i = 0; i < vectorSize; i++) {
        query[i] = dis(gen);
    }

    // run similarity search
    cout << "Running similarity search" << endl;
    for (bool cuda : {false, true}) {
        for (size_t batchSize : {1024, 2048, 4096, 8192, 16384, 32768, 65535 }) {
            steady_clock::time_point start = high_resolution_clock::now();
            vector<tuple<float, size_t>> result = findSimilar(
                flattenedEmbeddings,
                norms,
                query,
                numRows,
                vectorSize,
                topK,
                batchSize,
                cuda
            );
            steady_clock::time_point end = high_resolution_clock::now();

            // print time elapsed and other info
            duration<double> diff = end - start;
            double seconds = (double)diff.count();
            double eps = numRows / seconds;
            cout << (cuda ? "cuda" : "cpu") << ", batch size " << batchSize << ": "
                << seconds << " seconds total, " << eps << " embeddings/second" << endl;
            if (showResults) {
                cout << "results: [";
                for (size_t i = 0; i < result.size(); i++) {
                    const auto& [score, index] = result[i];
                    cout << "(" << score << "," << index << ")";
                    if (i < result.size() - 1) {
                        cout << ", ";
                    }
                }
                cout << "]" << endl;
            }
        }
    }

    return 0;
}
