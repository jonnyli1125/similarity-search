import argparse
import json
import glob

import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    model = get_model()
    embeddings = []
    file_lengths = []
    for filename in tqdm(glob.glob(f"{args.data_dir}/*.json")):
        with open(filename, "r", encoding="utf-8") as f:
            articles = json.load(f)
        file_lengths.append((filename, len(articles)))
        texts = [article["text"] for article in articles]
        new_embeddings = model.encode(texts, device="cuda", batch_size=1024, show_progress_bar=True)
        embeddings.append(new_embeddings)
    embeddings = np.concatenate(embeddings)
    print(embeddings.shape)
    np.save(f"{args.data_dir}/embeddings.npy", embeddings)
    with open(f"{args.data_dir}/file_lengths.txt", "w") as f:
        json.dump(file_lengths, f)
