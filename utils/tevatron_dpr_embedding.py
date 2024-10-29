import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from contextlib import nullcontext

import numpy as np
import torch
from datasets import Dataset
from tevatron.data import EncodeCollator, EncodeDataset
from tevatron.modeling import DenseModelForInference
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from utils.models.embedding_model import EmbeddingModel


class TevatronDPRCorpusEmbedding(EmbeddingModel):
    __name__ = "TevatronDPRCorpusEmbedding"

    def __init__(self, model: str, name=None, max_length=128, batch_size=8):
        if name:
            self.__name__ = name

        num_labels = 1
        self.config = AutoConfig.from_pretrained(
            model,
            num_labels=num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=False,
        )
        self.model = DenseModelForInference.build(model, config=self.config)
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.model.eval()

    def __call__(self, texts: list[str]):
        corpus_data = Dataset.from_list(
            [{"text_id": i, "text": x} for i, x in enumerate(texts)]
        )
        corpus_dataset = EncodeDataset(
            dataset=corpus_data, tokenizer=self.tokenizer, max_len=self.max_length
        )

        corpus_encode_loader = DataLoader(
            corpus_dataset,
            batch_size=self.batch_size,
            collate_fn=EncodeCollator(
                self.tokenizer, max_length=156, padding="max_length"
            ),
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )

        corpus_encoded = []

        for batch_ids, batch in tqdm(corpus_encode_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                with torch.no_grad():
                    model_output = self.model(passage=batch)
                    corpus_encoded.append(model_output.p_reps.cpu().detach().numpy())

        corpus_encoded = np.concatenate(corpus_encoded)

        return corpus_encoded


class TevatronDPRQueryEmbedding(EmbeddingModel):
    __name__ = "TevatronDPRQueryEmbedding"

    def __init__(self, model: str, name=None, max_length=32, batch_size=8):
        if name:
            self.__name__ = name

        num_labels = 1
        self.config = AutoConfig.from_pretrained(
            model,
            num_labels=num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=False,
        )
        self.model = DenseModelForInference.build(model, config=self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.max_length = max_length
        self.batch_size = batch_size

        self.model.eval()

    def __call__(self, texts: list[str]):
        query_data = Dataset.from_list(
            [{"text_id": i, "text": x} for i, x in enumerate(texts)]
        )
        query_dataset = EncodeDataset(
            dataset=query_data, tokenizer=self.tokenizer, max_len=self.max_length
        )

        query_encode_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            collate_fn=EncodeCollator(
                self.tokenizer, max_length=self.max_length, padding="max_length"
            ),
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )

        query_encoded = []

        for batch_ids, batch in tqdm(query_encode_loader):
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    model_output = self.model(query=batch)
                    query_encoded.append(model_output.q_reps.cpu().detach().numpy())

        query_encoded = np.concatenate(query_encoded)

        return query_encoded


if __name__ == "__main__":
    dpr_corpus_embedding = TevatronDPRCorpusEmbedding(model="models/model_finbert_nq")
    dpr_query_embedding = TevatronDPRQueryEmbedding(model="models/model_finbert_nq")

    print(dpr_corpus_embedding(["Your text string goes here", "Your text is awesome!"]))
    print(dpr_query_embedding(["Your text string goes here", "Your text is awesome!"]))
