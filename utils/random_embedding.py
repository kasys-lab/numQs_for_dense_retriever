import numpy as np
from tqdm import tqdm

from utils.models.embedding_model import EmbeddingModel


class RandomEmbedding(EmbeddingModel):
    __name__ = "RandomEmbedding"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, texts: list[str]):
        embeddings = []
        for i in tqdm(range(0, len(texts), 1024)):
            embeddings.append(np.random.rand(len(texts[i : i + 1024]), self.dim))

        embeddings = np.concatenate(embeddings)
        embeddings = embeddings.astype(np.float32)

        return embeddings
