import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from utils.models.embedding_model import EmbeddingModel

load_dotenv()

client = OpenAI()


class OpenAIEmbedding(EmbeddingModel):
    __name__ = "OpenAIEmbedding"

    def __init__(self, model="text-embedding-3-small"):
        self.model = model

    def __call__(self, texts: list[str]):
        max_batch = 1024

        embeddings = []
        for i in range(0, len(texts), max_batch):
            response = client.embeddings.create(
                input=texts[i : i + max_batch], model=self.model
            )

            batch_embeddings = list(map(lambda x: x.embedding, response.data))
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        return embeddings


if __name__ == "__main__":
    openai_embedding = OpenAIEmbedding()

    print(openai_embedding(["Your text string goes here", "Your text is awesome!"]))
