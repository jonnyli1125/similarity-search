import argparse
import json
import functools
import sys

from embeddings import get_model
from index import IVFIndex
from utils import latency


class DocsDB:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(f"{data_dir}/file_lengths.json", "r") as f:
            file_lengths = json.load(f)
        self.idx2file = []
        for filename, num_articles in file_lengths:
            self.idx2file += [(filename, i) for i in range(num_articles)]

    @functools.lru_cache
    def read_file(self, filename):
        with open(f"{self.data_dir}/{filename}", "r", encoding="utf-8") as f:
            return json.load(f)

    @functools.lru_cache
    def get(self, idx):
        filename, offset = self.idx2file[idx]
        data = self.read_file(filename)
        return data[offset]["text"]

class DocsIndex:
    def __init__(self, model, embed_index, docs_db):
        self.model = model
        self.embed_index = embed_index
        self.docs_db = docs_db

    def search(self, query_text, k):
        query = self.model.encode(query_text)
        results = self.embed_index.search(query, k, use_cuda=True)
        return [(score, self.docs_db.get(i)) for score, i in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    print("Loading...")
    model = get_model()
    embed_index = IVFIndex.from_pretrained(args.data_dir)
    docs_db = DocsDB(args.data_dir)
    docs_index = DocsIndex(model, embed_index, docs_db)
    print("Ready. Type any query:")

    for query_text in sys.stdin:
        results, seconds = latency(docs_index.search, query_text, k=3)
        for i, (score, doc_text) in enumerate(results):
            print(f"{i+1} ({score}): {doc_text[:100]}...")
        print(seconds, "seconds")
