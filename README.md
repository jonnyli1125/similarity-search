# Vector Similarity Search Engine for RAG

This project implements GPU-accelerated similarity search using C++, CUDA, and Python. It's kind of like the [faiss](https://github.com/facebookresearch/faiss) library, but I implemented things from scratch for learning purposes.

Features include:
- Custom CUDA kernel to perform cosine similarity between a single query embedding and a batch of embeddings
- C++ library that interfaces with numpy through Pybind to perform the search and call the CUDA kernels
- Python library that implements different embedding indexes with numpy/scikit-learn and RAG pipelines

# Usage

First, you need to build the C++/CUDA library. You need to have CUDA and pybind11 installed.
```bash
cmake -B build -S cpp
cmake --build build
cmake --install build
```

I am using [Plain Text Wikipedia from Kaggle](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011/data), which contains ~6 million text articles. To make things simple, 1 article = 1 embedding. I downloaded and extracted the data to `$DATA_DIR`.

To create embedding vectors and clusterings for IVF index:
```bash
cd python
python3 embeddings.py $DATA_DIR
python3 clusters.py $DATA_DIR --nlist 128
```

Run similarity search with different indexes and measure the time elapsed:
```bash
python3 index.py $DATA_DIR --index flat
python3 index.py $DATA_DIR --index ivf
```

Run RAG pipeline:
```bash
python3 rag.py $DATA_DIR
```

# Speed Comparison

Search time for top 10 similar embeddings to the query (see [index.py](./python/index.py) for details):

| Index Type                 | Device | Time    |
| -------------------------- | ------ | ------- |
| Flat                       | CPU    | 7.62s   |
| Flat                       | CUDA   | 3.01s   |
| IVF, clusters=128, probe=8 | CPU    | 1.30s   |
| IVF, clusters=128, probe=8 | CUDA   | 0.29s   |

Note: this is not a formal benchmark, I am just running it on my desktop PC.
