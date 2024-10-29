import os
from logging import getLogger

from ragatouille import RAGPretrainedModel

from utils.models.retriever_model import RetrieverModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logger = getLogger(__name__)


class ColBERTRetriever(RetrieverModel):
    __name__ = "ColBERT"
    doc_ids = None

    def __init__(self, model="colbert-ir/colbertv2.0", index_dir=None):
        self.retriever = RAGPretrainedModel.from_pretrained(model, index_root=index_dir)

    def fit(self, documents: list[str], ids: list[str]):
        self.retriever.index(
            index_name="colbert", collection=documents, document_ids=ids
        )

    def batch_search(self, queries: list[str], qids: list[str], top_k: int = 100):
        search_results = self.retriever.search(queries, k=top_k)

        result = {"search_results": search_results, "qids": qids}

        return result

    def save_as_trec(self, batch_result, output_path: str):
        with open(output_path, "w") as f:
            search_results = batch_result["search_results"]
            qids = batch_result["qids"]
            for query_id, search_result in zip(qids, search_results):
                for result in search_result:
                    f.write(
                        f"{query_id} Q0 {result['document_id']} {result['rank'] - 1} {result['score']} {self.__name__}\n"
                    )

    def search(self, query: str, top_k: int = 100):
        results = self.retriever.search(query=query, k=top_k)
        return results

    def __call__(self, query: str, top_k: int = 100):
        return self.search(query, top_k)


if __name__ == "__main__":
    colbert_retriever = ColBERTRetriever(
        max_context_length=64, nbits=1, kmeans_niters=4
    )

    documents = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog and the brown dog barks",
        "The brown dog barks at the quick fox",
        "The cat is sleeping on the couch",
        "The cat is sleeping on the couch and the dog is barking",
        "The dog is barking at the cat",
        "The rabbit is hopping around the garden",
        "The rabbit is hopping around the garden and the squirrel is watching",
        "The squirrel is watching the rabbit",
        "The bird is flying in the sky",
        "The bird is flying in the sky and the fish is swimming",
        "The fish is swimming in the water",
    ]

    ids = [f"doc{i}" for i in range(len(documents))]

    colbert_retriever.fit(documents, ids)

    print("ready to search")

    queries = ["quick brown fox", "brown dog barks", "quick fox"]

    print(colbert_retriever("quick brown fox"))
