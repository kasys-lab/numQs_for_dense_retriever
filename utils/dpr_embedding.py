from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from utils.models.embedding_model import EmbeddingModel


class DPRCorpusEmbedding(EmbeddingModel):
    __name__ = "DPRCorpusEmbedding"

    def __init__(self, model="facebook/dpr-ctx_encoder-single-nq-base", batch_size=8):
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model)
        self.corpus_model = DPRContextEncoder.from_pretrained(model)
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.corpus_model = self.corpus_model.cuda()

    def __call__(self, texts: list[str]):
        dataloader = torch.utils.data.DataLoader(texts, batch_size=self.batch_size)
        encoded = []
        for batch in tqdm(dataloader):
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                with torch.no_grad():
                    inputs_ids = self.tokenizer(
                        batch, return_tensors="pt", padding=True, truncation=True
                    )["input_ids"]

                    if torch.cuda.is_available():
                        inputs_ids = inputs_ids.cuda()

                    model_output = self.corpus_model(inputs_ids)
                    encoded.append(model_output.pooler_output.cpu().detach().numpy())

        encoded = np.concatenate(encoded)

        return encoded


class DPRQueryEmbedding(EmbeddingModel):
    __name__ = "DPRQueryEmbedding"

    def __init__(
        self, model="facebook/dpr-question_encoder-single-nq-base", batch_size=8
    ):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model)
        self.query_model = DPRQuestionEncoder.from_pretrained(model)
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.query_model = self.query_model.cuda()

    def __call__(self, texts: list[str]):
        dataloader = torch.utils.data.DataLoader(texts, batch_size=self.batch_size)
        encoded = []
        for batch in tqdm(dataloader):
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                with torch.no_grad():
                    inputs_ids = self.tokenizer(
                        batch, return_tensors="pt", padding=True, truncation=True
                    )["input_ids"]

                    if torch.cuda.is_available():
                        inputs_ids = inputs_ids.cuda()

                    model_output = self.query_model(inputs_ids)
                    encoded.append(model_output.pooler_output.cpu().detach().numpy())

        encoded = np.concatenate(encoded)

        return encoded
