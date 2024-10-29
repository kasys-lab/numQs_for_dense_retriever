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
from experiments.synthetic_unit_dataset.make_unit_datasets import UnitDatasetGenerator
from experiments.synthetic_unit_dataset.unit_dataset_benchmark import (
    ArtificialUnitDataset,
)

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
    num_times = cfg.dataset.num_times
    num_range = cfg.dataset.num_range
    num_step = cfg.dataset.num_step
    seed = cfg.dataset.seed

    generator_names, data_filepahts, query_patterns_filepahts = (
        UnitDatasetGenerator.generate_csv(
            output_dir, num_range, num_step, num_times, seed
        )
    )

    ### Benchmark
    depth = cfg.benchmark.depth
    query_target_category = ["none"]
    max_corpus = cfg.benchmark.max_corpus
    query_num = -1
    query_nums_pick_replace = False
    only_save_relevance = cfg.benchmark.only_save_relevance
    measures = cfg.benchmark.measures
    execlude_non_numeric = cfg.benchmark.execlude_non_numeric
    execlude_only_numeric = cfg.benchmark.execlude_only_numeric

    entity_category_primary_column = "name"  # not used
    entity_category_secondary_column = "name"  # not used
    number_column = "number"

    for generator_name, data_filepath, query_pattern_filepath in zip(
        generator_names, data_filepahts, query_patterns_filepahts
    ):
        benchmark_output_dir = f"{output_dir}/{generator_name}"
        os.makedirs(benchmark_output_dir, exist_ok=True)

        benchmark_output_figure_dir = f"{benchmark_output_dir}/figures"
        os.makedirs(benchmark_output_figure_dir, exist_ok=True)

        data_generator = ArtificialUnitDataset(
            data_filepath,
            entity_category_primary_column,
            entity_category_secondary_column,
            number_column,
            max_corpus=max_corpus,
            query_num=query_num,
            only_save_relevance=only_save_relevance,
            query_nums_pick_replace=query_nums_pick_replace,
            query_pattern_filepath=query_pattern_filepath,
            query_num_range=num_range,
            query_num_step=num_step,
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
    main()
    main()
