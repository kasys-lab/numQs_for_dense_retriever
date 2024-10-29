from abc import ABC


class RetrieverModel(ABC):
    __name__ = "SparseRetrieverModel"

    def __init__(self):
        pass

    def fit(self, documents: list[str], ids: list[str]):
        raise NotImplementedError()

    def batch_search(self, queries: list[str], qids: list[str], top_k: int = 100):
        raise NotImplementedError()

    def save_as_trec(self, batch_result, output_path: str):
        raise NotImplementedError()

    def search(self, query: str, top_k: int = 100):
        raise NotImplementedError()

    def __call__(self, query: str):
        return self.search(query)
