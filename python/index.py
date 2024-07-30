import heapq
import json

import numpy as np

from embeddings import get_model
from utils import latency, args_parser
import similarity_search


class FlatIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query, k, use_cuda=False):
        return similarity_search.find_similar(self.embeddings, query, k, use_cuda=use_cuda)

    @staticmethod
    def from_pretrained(data_dir):
        embeddings = np.load(f"{data_dir}/embeddings.npy")
        print("embeddings:", embeddings.shape)
        return FlatIndex(embeddings)


class IVFIndex:
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

    @staticmethod
    def from_pretrained(data_dir, n_probe=8):
        # load files
        with open(f"{data_dir}/cluster_mappings.json", "r") as f:
            cluster_mappings = json.load(f)
        n_clusters = len(cluster_mappings)
        cluster_embeddings = [np.load(f"{data_dir}/cluster_embeddings_{i}.npy") for i in range(n_clusters)]
        cluster_centroids = np.load(f"{data_dir}/cluster_centroids.npy")

        # print info
        n_embeddings = sum(c.shape[0] for c in cluster_embeddings)
        embed_dim = cluster_centroids.shape[1]
        print("embeddings:", (n_embeddings, embed_dim))
        print("centroids:", cluster_centroids.shape)

        return IVFIndex(cluster_embeddings, cluster_mappings, cluster_centroids, n_probe=n_probe)


if __name__ == "__main__":
    parser = args_parser()
    parser.add_argument("-i", "--index", choices=["flat", "ivf"], default="flat")
    parser.add_argument("-r", "--result", action="store_true", default=False)
    args = parser.parse_args()

    print("Loading embeddings index")
    if args.index == "flat":
        index = FlatIndex.from_pretrained(args.data_dir)
    elif args.index == "ivf":
        index = IVFIndex.from_pretrained(args.data_dir)
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
