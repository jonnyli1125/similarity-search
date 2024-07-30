from openai import OpenAI

from docs import DocsIndex
from utils import latency, args_parser


class RAG:
    def __init__(self, docs_index):
        self.openai = OpenAI()
        self.docs_index = docs_index

    def format_query(self, query_text, similar_results):
        message = "=== BEGIN DOCUMENTS ===\n"
        for _, doc_text in similar_results:
            message += doc_text + "\n\n"
        message += "=== END DOCUMENTS ===\n\n" + query_text
        return message

    def generate(self, query_text, k=3):
        similar_results = self.docs_index.search(query_text, k=k)
        message = self.format_query(query_text, similar_results)
        messages = [{"role": "user", "content": message}]
        return self.openai.chat.completions.create(messages=messages, model="gpt-4o").choices[0].message.content

    @staticmethod
    def from_pretrained(data_dir):
        docs_index = DocsIndex.from_pretrained(data_dir)
        return RAG(docs_index)

if __name__ == "__main__":
    args = args_parser().parse_args()

    print("Loading...")
    rag = RAG.from_pretrained(args.data_dir)
    print("Ready. Type any query:")
    while True:
        query_text = input("> ")
        response, seconds = latency(rag.generate, query_text)
        print(response)
        print(seconds, "seconds")
