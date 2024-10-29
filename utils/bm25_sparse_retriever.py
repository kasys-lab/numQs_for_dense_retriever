import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import pandas as pd
import pyterrier as pt

from utils.models.retriever_model import RetrieverModel


class BM25SparseRetriever(RetrieverModel):
    __name__ = "BM25"
    indexer = None
    bm25 = None
    index_dir = "../result/bm25"

    def __init__(self, index_dir: str = "../result/bm25"):
        super().__init__()

        if not pt.started():
            pt.init()  # type: ignore

        if os.path.exists(index_dir):
            for file in os.listdir(index_dir):
                os.remove(os.path.join(index_dir, file))

        self.index_dir = index_dir

    def fit(self, documents: list[str], ids: list[str]):
        if os.path.exists(self.index_dir):
            for file in os.listdir(self.index_dir):
                os.remove(os.path.join(self.index_dir, file))

        document_df = pd.DataFrame({"text": documents, "docno": ids})
        document_df["text"] = document_df["text"].apply(self.sanitize_symbol)

        self.indexer = pt.DFIndexer(
            self.index_dir,
            verbose=True,
            sttemmer="none",
            stopwords=None,
            tokeniser="whitespace",
        )
        # self.indexer.setProperty("tokeniser", "IdentityTokeniser")
        self.indexer.index(document_df["text"], document_df["docno"])  # type: ignore
        self.bm25 = pt.BatchRetrieve(self.indexer, wmodel="BM25")

    def batch_search(self, queries: list[str], qids: list[str], top_k: int = 100):
        query_df = pd.DataFrame({"query": queries, "qid": qids})

        # sanitize query
        query_df["query"] = query_df["query"].apply(self.sanitize_symbol)

        pipeline = pt.BatchRetrieve(self.indexer, wmodel="BM25")

        result = pipeline.transform(query_df).groupby("qid").head(top_k)
        return result

    @staticmethod
    def sanitize_symbol(query: str):
        return (
            query.strip()
            .replace("'", "")
            .replace("?", "")
            .replace("!", "")
            .replace(".", "")
            .replace(",", "")
            .replace(":", "")
            .replace(";", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace("{", "")
            .replace("}", "")
            .replace("/", "")
        )

    def save_as_trec(self, batch_result, output_path: str):
        pt.io.write_results(batch_result, output_path, format="trec")

    def search(self, query: str, top_k: int = 100):
        return self.bm25.search(query, top_k)  # type: ignore
