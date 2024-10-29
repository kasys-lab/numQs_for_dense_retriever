from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.models.embedding_model import EmbeddingModel


class E5Embedding(EmbeddingModel):
    @property
    def __name__(self):
        return "E5Embedding" + self.suffix

    def __init__(
        self, model="intfloat/e5-base-v2", mode="passage", suffix="", batch_size=8
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.mode = mode
        self.suffix = suffix
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __call__(self, texts: list[str]):
        if self.mode == "passage":
            texts = [f"passage: {text}" for text in texts]
        elif self.mode == "query":
            texts = [f"query: {text}" for text in texts]
        else:
            raise ValueError("mode must be either 'passage' or 'query'")

        dataloader = torch.utils.data.DataLoader(texts, batch_size=self.batch_size)

        encoded = []
        for batch in tqdm(dataloader):
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                with torch.no_grad():
                    batch_dict = self.tokenizer(
                        batch,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )

                    if torch.cuda.is_available():
                        for key in batch_dict.keys():
                            batch_dict[key] = batch_dict[key].cuda()

                    outputs = self.model(**batch_dict)
                    embeddings = self.average_pool(
                        outputs.last_hidden_state, batch_dict["attention_mask"]
                    )

                    encoded.append(embeddings.cpu().detach().numpy())

        encoded = np.concatenate(encoded)

        return encoded
