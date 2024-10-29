import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from experimental_settings.numerical_condition_benchmark_base import (
    NumericalConditionBenchmarkBase,
)
from experiments.movie_dataset.make_movie_datasets import create_movie_datasets
from experiments.movie_dataset.movie_revenue_benchmark import MovieRevenueDataset

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    base_output_dir = HydraConfig.get().runtime.output_dir
    output_dir = os.path.join(base_output_dir, "experiment_files")
    logger.info(f"output dir here: {base_output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = cfg.encoder.batch_size

    encoders = []

    if cfg.encoder.dpr:
        from utils.dpr_embedding import DPRCorpusEmbedding, DPRQueryEmbedding

        dpr_models = cfg.encoder.dpr.models

        for model in dpr_models:
            query_model = model.query_model
            corpus_model = model.corpus_model
            name = model.name
            encoders.append(
                (
                    DPRCorpusEmbedding(model=corpus_model, batch_size=batch_size),
                    DPRQueryEmbedding(model=query_model, batch_size=batch_size),
                    name,
                )
            )

    if cfg.encoder.tevatron_dpr:
        from utils.tevatron_dpr_embedding import (
            TevatronDPRCorpusEmbedding,
            TevatronDPRQueryEmbedding,
        )

        tevatron_models = cfg.encoder.tevatron_dpr.models

        for model in tevatron_models:
            query_model = model.query_model
            corpus_model = model.corpus_model
            name = model.name
            encoders.append(
                (
                    TevatronDPRCorpusEmbedding(
                        model=corpus_model, name=name, batch_size=batch_size
                    ),
                    TevatronDPRQueryEmbedding(
                        model=query_model, name=name, batch_size=batch_size
                    ),
                    name,
                )
            )

    if cfg.encoder.e5:
        from utils.e5_embedding import E5Embedding

        e5_models = cfg.encoder.e5.models

        for model in e5_models:
            model_name = model.model
            suffix = model.suffix
            name = model.name
            encoders.append(
                (
                    E5Embedding(
                        model=model_name,
                        mode="passage",
                        suffix=suffix,
                        batch_size=batch_size,
                    ),
                    E5Embedding(
                        model=model_name,
                        mode="query",
                        suffix=suffix,
                        batch_size=batch_size,
                    ),
                    name,
                )
            )

    if cfg.encoder.random:
        from utils.random_embedding import RandomEmbedding

        encoders.append((RandomEmbedding(), RandomEmbedding(), "Random"))

    if cfg.encoder.openai:
        from utils.openai_embedding import OpenAIEmbedding

        openai_models = cfg.encoder.openai.models

        for model in openai_models:
            model_name = model.model
            name = model.name

            encoders.append(
                (
                    OpenAIEmbedding(model=model_name),
                    OpenAIEmbedding(model=model_name),
                    name,
                )
            )

    ### Other retrievers
    other_retrievers = []

    if cfg.retriever.bm25:
        from utils.bm25_sparse_retriever import BM25SparseRetriever

        bm25_index_path = f"{output_dir}/bm25_index"
        other_retrievers.append(BM25SparseRetriever(index_dir=bm25_index_path))

    if cfg.retriever.colbert:
        from utils.colbert_retriever import ColBERTRetriever

        colbert_model = cfg.retriever.colbert.model
        colbert_index_path = f"{output_dir}/colbert_index"
        other_retrievers.append(
            ColBERTRetriever(model=colbert_model, index_dir=colbert_index_path)
        )

    if len(encoders) == 0 and len(other_retrievers) == 0:
        raise ValueError("No encoders or retrievers are specified.")

    ### dataset
    raw_data_filepath = cfg.dataset.raw_data_filepath

    create_movie_datasets(output_dir, raw_data_filepath)

    movie_dataset_path = f"{output_dir}/movie_dataset.csv"
    shuffled_movie_dataset_path = f"{output_dir}/shuffled_movie_dataset.csv"

    ### Benchmark
    depth = cfg.benchmark.depth
    query_target_category = cfg.benchmark.query_target_category
    max_corpus = cfg.benchmark.max_corpus
    query_num = cfg.benchmark.query_num
    query_nums_pick_replace = cfg.benchmark.query_nums_pick_replace
    only_save_relevance = cfg.benchmark.only_save_relevance
    measures = cfg.benchmark.measures
    execlude_non_numeric = cfg.benchmark.execlude_non_numeric
    execlude_only_numeric = cfg.benchmark.execlude_only_numeric

    revenue_columns = cfg.benchmark.revenue_columns
    currencies = cfg.benchmark.currencies
    is_exchange_rate_list = cfg.benchmark.is_exchange_rate_list
    movie_dataset_types = cfg.benchmark.movie_dataset_types

    entity_category_primary_column = "genres"
    entity_category_secondary_column = "production_companies"

    movie_dataset_paths = {
        "original": movie_dataset_path,
        "shuffled": shuffled_movie_dataset_path,
    }

    number_modes = {
        "revenue_int": "none",
        "revenue_int_k": "K",
        "revenue_int_m": "M",
    }

    for revenue_column in revenue_columns:
        for currency in currencies:
            for is_exchange_rate in is_exchange_rate_list:
                if is_exchange_rate and currency == "dollars":
                    continue

                for movie_dataset_type in movie_dataset_types:
                    benchmark_output_dir = f"{output_dir}/benchmark_{movie_dataset_type}_{revenue_column}_{currency}_{'exchange_rate' if is_exchange_rate else 'original'}"
                    os.makedirs(benchmark_output_dir, exist_ok=True)

                    benchmark_output_figure_dir = f"{benchmark_output_dir}/figures"
                    os.makedirs(benchmark_output_figure_dir, exist_ok=True)

                    number_mode = number_modes[revenue_column]
                    dataset_path = movie_dataset_paths[movie_dataset_type]
                    data_generator = MovieRevenueDataset(
                        dataset_path,
                        entity_category_primary_column,
                        entity_category_secondary_column,
                        revenue_column,
                        number_mode=number_mode,
                        currency=currency,
                        max_corpus=max_corpus,
                        query_num=query_num,
                        only_save_relevance=only_save_relevance,
                        is_exchange_rate=is_exchange_rate,
                        query_nums_pick_replace=query_nums_pick_replace,
                    )

                    benchmark = NumericalConditionBenchmarkBase(
                        encoders=encoders,
                        other_retrievers=other_retrievers,
                        generator=data_generator,
                        query_target_category=query_target_category,
                        depth=depth,
                    )

                    benchmark.benchmark_every_operator(
                        benchmark_output_dir,
                        measures=measures,
                        execlude_non_numeric=execlude_non_numeric,
                        execlude_only_numeric=execlude_only_numeric,
                    )

                    logger.info("Benchmark finished.")


if __name__ == "__main__":
    main()
    main()
