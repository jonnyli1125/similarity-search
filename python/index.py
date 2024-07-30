import argparse
import heapq
import json
from typing import Protocol

import numpy as np

from embeddings import get_model
from utils import latency
import similarity_search


class Index(Protocol):
    def search(self, query, k, use_cuda=False):
        pass


class FlatIndex(Index):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query, k, use_cuda=False):
        return similarity_search.find_similar(self.embeddings, query, k, use_cuda=use_cuda)


class IVFIndex(Index):
    def __init__(self, cluster_embeddings, cluster_mappings, cluster_centroids, n_probe):
        self.n_probe = n_probe
        self.cluster_embeddings = cluster_embeddings
        self.cluster_mappings = cluster_mappings
        self.cluster_centroids = cluster_centroids

    def search(self, query, k, use_cuda=False):
        top_centroids = similarity_search.find_similar(self.cluster_centroids, query, self.n_probe, use_cuda=use_cuda)
        search_iter = (
            (score, self.cluster_mappings[cluster][idx])
            for _, cluster in top_centroids
            for score, idx in similarity_search.find_similar(self.cluster_embeddings[cluster], query, k, use_cuda=use_cuda)
        )
        return heapq.nlargest(k, search_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("-i", "--index", choices=["flat", "ivf"], default="flat")
    parser.add_argument("-r", "--result", action="store_true", default=False)
    args = parser.parse_args()

    print("Loading embeddings index")
    if args.index == "flat":
        embeddings = np.load(f"{args.data_dir}/embeddings.npy")
        print("embeddings:", embeddings.shape)
        index = FlatIndex(embeddings)
    elif args.index == "ivf":
        # load files
        with open(f"{args.data_dir}/cluster_mappings.json", "r") as f:
            cluster_mappings = json.load(f)
        n_clusters = len(cluster_mappings)
        cluster_embeddings = [np.load(f"{args.data_dir}/cluster_embeddings_{i}.npy") for i in range(n_clusters)]
        cluster_centroids = np.load(f"{args.data_dir}/cluster_centroids.npy")

        # print info
        n_embeddings = sum(c.shape[0] for c in cluster_embeddings)
        embed_dim = cluster_centroids.shape[1]
        print("clusters:", n_clusters)
        print("embeddings:", (n_embeddings, embed_dim))
        print("centroids:", cluster_centroids.shape)

        # create index
        index = IVFIndex(cluster_embeddings, cluster_mappings, cluster_centroids, n_probe=8)
    else:
        raise ValueError("Invalid embeddings index type")

    query = get_model().encode("What is deep learning?")
    print("query:", query.shape)
    for use_cuda in [False, True]:
        name = "cuda" if use_cuda else "cpu"
        result, seconds = latency(index.search, query, k=10, use_cuda=use_cuda)
        print(f'{name}: {seconds}')
        if args.result:
            print(result)
