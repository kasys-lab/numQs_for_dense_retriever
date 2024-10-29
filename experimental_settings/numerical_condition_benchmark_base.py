# type: ignore
import ast
import csv
import json
import logging
import os
import pickle
import random
import subprocess
from abc import ABC
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.models.embedding_model import EmbeddingModel
from utils.models.retriever_model import RetrieverModel


class NumericalConditionDatasetBuilderBase(ABC):
    PRIMARY_CATEGORY_DICT_NAME = "primary"
    SECONDARY_CATEGORY_DICT_NAME = "secondary"

    def __init__(
        self,
        csv_data_filepath,
        entity_category_primary_column,
        entity_category_secondary_column,
        number_column,
        max_corpus=-1,
        query_num=5000,
        only_save_relevance=True,
        is_primary_category_list=False,
        is_secondary_category_list=False,
        seed=42,
        query_nums_pick_replace=True,
    ):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.csv_data_filepath = csv_data_filepath
        self.entity_category_primary_column = entity_category_primary_column
        self.entity_category_secondary_column = entity_category_secondary_column
        self.number_column = number_column
        self.only_save_relevance = only_save_relevance
        self.max_corpus = max_corpus

        self.is_primary_category_list = is_primary_category_list
        self.is_secondary_category_list = is_secondary_category_list

        self.load_csv_data()

        self.query_num_pick_replace = query_nums_pick_replace
        self.query_num = query_num
        self.pickup_query_numbers(query_num)

        self.primary_entity_categories = self.get_primary_entity_categories()
        self.secondary_entity_categories = self.get_secondary_entity_categories()

    def load_csv_data(self):
        df = pd.read_csv(self.csv_data_filepath)

        self.raw_data_df = df

        if self.is_primary_category_list:
            self.raw_data_df[self.entity_category_primary_column] = self.raw_data_df[
                self.entity_category_primary_column
            ].apply(lambda x: ast.literal_eval(x))

        if self.is_secondary_category_list:
            self.raw_data_df[self.entity_category_secondary_column] = self.raw_data_df[
                self.entity_category_secondary_column
            ].apply(lambda x: ast.literal_eval(x))

    @property
    def raw_data_length(self):
        return len(self.raw_data_df)

    def get_primary_entity_categories(self):
        if self.is_primary_category_list:
            return (
                self.raw_data_df[self.entity_category_primary_column].explode().unique()
            )
        else:
            return self.raw_data_df[self.entity_category_primary_column].unique()

    def get_secondary_entity_categories(self):
        if self.is_secondary_category_list:
            return (
                self.raw_data_df[self.entity_category_secondary_column]
                .explode()
                .unique()
            )
        else:
            return self.raw_data_df[self.entity_category_secondary_column].unique()

    def pickup_query_numbers(self, query_num):
        numbers = self.raw_data_df[self.number_column].unique()

        if self.query_num_pick_replace:
            self.query_numbers = np.random.choice(numbers, query_num).tolist()
            self.query_numbers_secondary = np.random.choice(numbers, query_num).tolist()
        else:
            if query_num > len(numbers):
                logging.warning("query_num is larger than the number of unique numbers")
                logging.warning("query_num is set to the number of unique numbers")
                query_num = len(numbers)
                self.query_num = query_num

            self.query_numbers = np.random.choice(
                numbers, query_num, replace=False
            ).tolist()
            self.query_numbers_secondary = np.random.choice(
                numbers, query_num, replace=False
            ).tolist()

    def generate_corpus_text(
        self, primary_category, secondary_category, number, metadata=None
    ):
        raise NotImplementedError

    def generate_corpus(
        self, primary_category, secondary_category, number, metadata=None
    ):
        text = self.generate_corpus_text(
            primary_category, secondary_category, number, metadata
        )

        entities = {
            self.PRIMARY_CATEGORY_DICT_NAME: primary_category,
            self.SECONDARY_CATEGORY_DICT_NAME: secondary_category,
            "number": int(number),
            "metadata": metadata,
        }

        data = {
            "text": text,
            "entities": entities,
            "number": int(number),
        }

        return data

    def generate_corpus_list(self):
        corpus_list = []

        if self.max_corpus > 0 and self.max_corpus < self.raw_data_length:
            data_index_iter = np.random.choice(
                self.raw_data_length, self.max_corpus, replace=False
            )  # replace=False is no duplication
        else:
            data_index_iter = range(self.raw_data_length)

        for i in data_index_iter:
            primary_category = self.raw_data_df.iloc[i][
                self.entity_category_primary_column
            ]
            secondary_category = self.raw_data_df.iloc[i][
                self.entity_category_secondary_column
            ]
            number = self.raw_data_df[self.number_column][i]

            metadata = self.raw_data_df.iloc[i].to_dict()

            corpus = self.generate_corpus(
                primary_category, secondary_category, number, metadata
            )
            corpus["id"] = f"corpus_{i}"

            corpus_list.append(corpus)

        return corpus_list

    def generate_primary_query_target(self, primary_category):
        raise NotImplementedError

    def generate_secondary_query_target(self, secondary_category):
        raise NotImplementedError

    def generate_primary_secondary_query_target(
        self, primary_category, secondary_category
    ):
        raise NotImplementedError

    def generate_none_query_target(self):
        raise NotImplementedError

    def generate_equal_query_text(self, query_target, query_number):
        raise NotImplementedError

    def generate_less_query_text(self, query_target, query_number):
        raise NotImplementedError

    def generate_more_query_text(self, query_target, query_number):
        raise NotImplementedError

    def generate_between_query_text(self, query_target, query_number):
        raise NotImplementedError

    def generate_around_query_text(self, query_target, query_number):
        raise NotImplementedError

    def generate_none_query_text(self, query_target):
        raise NotImplementedError

    def generate_query(
        self,
        query_operator_category,
        query_target_category,
        query_value: Union[int, List[int]],
        preset_filter_entities=None,
    ):
        query_operator_categories = [
            "equal",
            "less",
            "more",
            "between",
            "around",
            "none",
        ]

        if query_operator_category not in query_operator_categories:
            raise ValueError(
                "query_operator must be either 'equal', 'less', 'more', 'between', 'around', or 'none'"
            )

        query_target_categories = ["primary", "secondary", "primary_secondary", "none"]

        if query_target_category not in query_target_categories:
            raise ValueError(
                "query_target must be either 'primary', 'secondary', or 'primary_secondary'"
            )

        filter_entities = {}
        if query_target_category == "primary":
            if not preset_filter_entities:
                primary_category = random.choice(self.primary_entity_categories)
            else:
                primary_category = preset_filter_entities[
                    self.PRIMARY_CATEGORY_DICT_NAME
                ]

            query_target = self.generate_primary_query_target(primary_category)

            filter_entities[self.PRIMARY_CATEGORY_DICT_NAME] = primary_category
        elif query_target_category == "secondary":
            if not preset_filter_entities:
                secondary_category = random.choice(self.secondary_entity_categories)
            else:
                secondary_category = preset_filter_entities[
                    self.SECONDARY_CATEGORY_DICT_NAME
                ]

            query_target = self.generate_secondary_query_target(secondary_category)

            filter_entities[self.SECONDARY_CATEGORY_DICT_NAME] = secondary_category
        elif query_target_category == "primary_secondary":
            if not preset_filter_entities:
                primary_category = random.choice(self.primary_entity_categories)
                secondary_category = random.choice(self.secondary_entity_categories)
            else:
                primary_category = preset_filter_entities[
                    self.PRIMARY_CATEGORY_DICT_NAME
                ]
                secondary_category = preset_filter_entities[
                    self.SECONDARY_CATEGORY_DICT_NAME
                ]

            query_target = self.generate_primary_secondary_query_target(
                primary_category, secondary_category
            )

            filter_entities[self.PRIMARY_CATEGORY_DICT_NAME] = primary_category
            filter_entities[self.SECONDARY_CATEGORY_DICT_NAME] = secondary_category
        elif query_target_category == "none":
            query_target = self.generate_none_query_target()
        else:
            raise ValueError("Invalid query target")

        if query_operator_category == "equal":
            query_text = self.generate_equal_query_text(query_target, query_value)
        elif query_operator_category == "less":
            query_text = self.generate_less_query_text(query_target, query_value)
        elif query_operator_category == "more":
            query_text = self.generate_more_query_text(query_target, query_value)
        elif query_operator_category == "between":
            query_text = self.generate_between_query_text(query_target, query_value)
        elif query_operator_category == "around":
            query_text = self.generate_around_query_text(query_target, query_value)
        elif query_operator_category == "none":
            query_text = self.generate_none_query_text(query_target)
        else:
            raise ValueError("Invalid query operator")

        data = {
            "text": query_text,
            "metadata": {
                "query_operator_category": query_operator_category,
                "query_target_category": query_target_category,
                "filter_entities": filter_entities,
                "query_value": query_value,
            },
        }

        return data

    def generate_query_list(
        self,
        query_operator_category,
        query_target_category: Union[str, List[str], None] = None,
    ):
        query_list = []

        if query_target_category is None:
            query_target_category = [
                "primary",
                "secondary",
                "primary_secondary",
                "none",
            ]

        if isinstance(query_target_category, str):
            query_target_category = [query_target_category]

        for i in range(self.query_num):
            selected_query_target_category = random.choice(query_target_category)

            if query_operator_category == "between":
                query_number = sorted(
                    [self.query_numbers[i], self.query_numbers_secondary[i]]
                )
            else:
                query_number = self.query_numbers[i]

            query = self.generate_query(
                query_operator_category, selected_query_target_category, query_number
            )

            query["id"] = f"query_{query_operator_category}_{i}"

            query_list.append(query)

        return query_list

    def generate_non_numeric_query_list(self, query_list):
        non_numeric_query_list = []

        for query in query_list:
            non_numeric_query = self.generate_query(
                "none",
                query["metadata"]["query_target_category"],
                query["metadata"]["query_value"],
                query["metadata"]["filter_entities"],
            )
            non_numeric_query["id"] = f"{query['id']}"

            non_numeric_query_list.append(non_numeric_query)

        return non_numeric_query_list

    def generate_only_numeric_query_list(self, query_list):
        only_numeric_query_list = []

        for query in query_list:
            query_operator_category = query["metadata"]["query_operator_category"]
            query_value = query["metadata"]["query_value"]
            only_numeric_query = self.generate_query(
                query_operator_category, "none", query_value
            )
            only_numeric_query["id"] = f"{query['id']}"

            only_numeric_query_list.append(only_numeric_query)

        return only_numeric_query_list

    def generate_qrels(self, query_list, corpus_list):
        qrels = []

        for query in tqdm(query_list):
            query_id = query["id"]
            query_value = query["metadata"]["query_value"]
            query_operator_category = query["metadata"]["query_operator_category"]
            filter_entities = query["metadata"]["filter_entities"]

            has_relevance_corpus_list = self.relevance_corpus(
                corpus_list, query_value, query_operator_category, filter_entities
            )

            query_qrels = []
            for index, corpus in enumerate(has_relevance_corpus_list):
                if self.only_save_relevance and corpus["relevance"] == 0:
                    continue

                qrels.append(
                    {
                        "qid": query_id,
                        "docid": corpus["id"],
                        "rel": 1,
                        "rank": index + 1,
                    }
                )

            qrels.extend(query_qrels)

        return qrels

    @staticmethod
    def is_match_corpus_entity(corpus, entities):
        for key, value in entities.items():
            if isinstance(corpus["entities"][key], list):
                if value not in corpus["entities"][key]:
                    return False
            else:
                if corpus["entities"][key] != value:
                    return False

        return True

    @staticmethod
    def relevance_corpus_operator(corpus, operator, value):
        relevance = 1
        if operator == "equal":
            if corpus["number"] == value:
                return relevance
        elif operator == "less":
            if corpus["number"] <= value:
                return relevance
        elif operator == "more":
            if corpus["number"] >= value:
                return relevance
        elif operator == "between":
            if not isinstance(value, list):
                raise ValueError("value must be a list on 'between' operator")

            if value[0] <= corpus["number"] <= value[1]:
                return relevance
        elif operator == "around":
            min_value = int(value * 0.85)
            max_value = int(value * 1.15)

            if min_value <= corpus["number"] <= max_value:
                return relevance
        elif operator == "none":
            return relevance
        else:
            raise ValueError(
                "operator must be either 'equal', 'less', 'more', 'between', 'around', or 'none'"
            )

        return 0

    @staticmethod
    def sort_corpus_by_relevance_operator(corpus_list, operator, value):
        result_corpus_list = corpus_list.copy()

        if result_corpus_list[0].get("relevance") is None:
            raise ValueError("relevance must be calculated before sorting")

        if operator == "equal":
            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: abs(x["number"] - value)
            )

            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: -x["relevance"]
            )

            return result_corpus_list
        elif operator == "less":
            result_corpus_list = sorted(result_corpus_list, key=lambda x: x["number"])

            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: -x["relevance"]
            )

            return result_corpus_list
        elif operator == "more":
            result_corpus_list = sorted(result_corpus_list, key=lambda x: -x["number"])

            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: -x["relevance"]
            )

            return result_corpus_list
        elif operator == "between":
            mid_value = (value[0] + value[1]) / 2
            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: abs(x["number"] - mid_value)
            )

            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: -x["relevance"]
            )

            return result_corpus_list
        elif operator == "around":
            result_corpus_list = sorted(
                result_corpus_list, key=lambda x: abs(x["number"] - value)
            )

            return result_corpus_list
        elif operator == "none":
            return result_corpus_list
        else:
            raise ValueError(
                "operator must be either 'equal', 'less', 'more', 'between', 'around', or 'none'"
            )

    def relevance_corpus(
        self, corpus_list, query_value, query_operator_category, filter_entites={}
    ):
        """
        Filter corpus list from the movie wikidata dataset
        """

        result_corpus_list = corpus_list.copy()
        for item in result_corpus_list:
            if self.is_match_corpus_entity(item, filter_entites):
                relevance = self.relevance_corpus_operator(
                    item, query_operator_category, query_value
                )
                item["relevance"] = relevance
            else:
                item["relevance"] = 0

        result_corpus_list = self.sort_corpus_by_relevance_operator(
            result_corpus_list, query_operator_category, query_value
        )

        return result_corpus_list


class NumericalConditionBenchmarkBase(ABC):
    def __init__(
        self,
        encoders: list[tuple[EmbeddingModel, EmbeddingModel, str]],
        other_retrievers: list[RetrieverModel],
        generator: NumericalConditionDatasetBuilderBase,
        query_target_category=None,
        depth=100,
    ):
        self.depth = depth
        self.query_target_category = query_target_category

        self.encoders = encoders
        self.other_retrievers = other_retrievers

        self.generator = generator

        self.corpus_list = self.generator.generate_corpus_list()

        corpus_texts = list(map(lambda x: x["text"], self.corpus_list))
        corpus_ids = list(map(lambda x: x["id"], self.corpus_list))

        for other_retriever in self.other_retrievers:
            other_retriever.fit(corpus_texts, corpus_ids)

    @staticmethod
    def make_query_embedding(query_list, query_encoder, save_dir, filename):
        query_texts = list(map(lambda x: x["text"], query_list))
        query_embeddings = query_encoder(query_texts)
        lookup_indices = list(map(lambda x: x["id"], query_list))

        with open(f"{save_dir}/{filename}", "wb") as f:
            pickle.dump((query_embeddings, lookup_indices), f)

    @staticmethod
    def make_corpus_embedding(corpus_list, corpus_encoder, save_dir, filename):
        corpus_texts = list(map(lambda x: x["text"], corpus_list))
        corpus_embeddings = corpus_encoder(corpus_texts)
        lookup_indices = list(map(lambda x: x["id"], corpus_list))

        with open(f"{save_dir}/{filename}", "wb") as f:
            pickle.dump((corpus_embeddings, lookup_indices), f)

    @staticmethod
    def faiss_retriever_with_subprocess(
        save_dir,
        query_emb="query_emb.pkl",
        corpus_emb="corpus_emb.pkl",
        result_filename="dpr",
        depth=100,
    ):
        logging.info("Running faiss_retriever with subprocess")

        text_filename = result_filename.replace(".trec", "") + ".txt"
        trec_filename = result_filename.replace(".trec", "") + ".trec"

        subprocess.run(
            [
                "python",
                "-m",
                "tevatron.faiss_retriever",
                "--query_reps",
                f"{save_dir}/{query_emb}",
                "--passage_reps",
                f"{save_dir}/{corpus_emb}",
                "--depth",
                str(depth),
                "--batch_size",
                "-1",
                "--save_text",
                "--save_ranking_to",
                f"{save_dir}/{text_filename}",
            ]
        )

        logging.info("Converting result to trec")

        subprocess.run(
            [
                "python",
                "-m",
                "tevatron.utils.format.convert_result_to_trec",
                "--input",
                f"{save_dir}/{text_filename}",
                "--output",
                f"{save_dir}/{trec_filename}",
            ]
        )

        os.remove(f"{save_dir}/{text_filename}")

    @staticmethod
    def measure_retrieval_score_with_subprocess(
        save_dir,
        qrels_file="qrels.tsv",
        trec_file="dpr.trec",
        result_file="measures.tsv",
        measures=["nDCG", "nDCG@100", "RR@100", "R@100"],
    ):
        logging.info("Measuring retrieval score with subprocess")

        base_commands = [
            "ir_measures",
            f"{save_dir}/{qrels_file}",
            f"{save_dir}/{trec_file}",
            "--output_format",
            "tsv",
        ]

        for measure in measures:
            base_commands.append(str(measure))

        ret = subprocess.run(base_commands, stdout=subprocess.PIPE, encoding="utf-8")

        logging.info(ret.stdout)

        with open(f"{save_dir}/{result_file}", "w") as f:
            f.write(ret.stdout)

        return ret.stdout

    @staticmethod
    def format_measure(result):
        measure_data = {}
        for line in result.split("\n"):
            if not line:
                continue

            item = line.strip().split("\t")
            measure, value = item[0], item[1]
            measure_data[measure] = float(value)

        return measure_data

    @staticmethod
    def check_number_correct(operator, target_number, qnumber):
        if operator == "equal":
            return target_number == qnumber
        elif operator == "less":
            return target_number <= qnumber
        elif operator == "more":
            return target_number >= qnumber
        elif operator == "around":
            low_number = int(qnumber * 0.85)
            high_number = int(qnumber * 1.15)

            return low_number <= target_number <= high_number
        elif operator == "between":
            if isinstance(qnumber, int):
                raise ValueError("Number should be list for between operator")

            low_number = qnumber[0]
            high_number = qnumber[1]

            return low_number <= target_number <= high_number
        elif operator == "none":
            return True
        else:
            raise ValueError("Invalid operator")

    @staticmethod
    def export_qrels(qrels, save_dir, filename="qrels.tsv"):
        with open(f"{save_dir}/{filename}", "w") as f:
            writer = csv.writer(f, delimiter="\t")
            for qrel in qrels:
                writer.writerow([qrel["qid"], "Q0", qrel["docid"], qrel["rel"]])

        return qrels

    @staticmethod
    def filter_numeric_rerank_trec(
        save_dir,
        query_list,
        corpus_list,
        trec_file="dpr.trec",
        filtered_trec_file="dpr_filtered.trec",
    ):
        query_dict = {query["id"]: query for query in query_list}
        corpus_dict = {corpus["id"]: corpus for corpus in corpus_list}

        with open(f"{save_dir}/{trec_file}", "r") as f, open(
            f"{save_dir}/{filtered_trec_file}", "w"
        ) as f2:
            for line in f:
                try:
                    qid, _, docid, rank, _, _ = line.strip().split(" ")

                    query = query_dict[qid]
                    corpus = corpus_dict[docid]

                    query_number = query["metadata"]["query_value"]

                    query_operator_category = query["metadata"][
                        "query_operator_category"
                    ]
                    corpus_number = corpus["number"]

                    if NumericalConditionBenchmarkBase.check_number_correct(
                        query_operator_category, corpus_number, query_number
                    ):
                        f2.write(line)
                except Exception as e:
                    logging.error(e)
                    logging.error(line)

    @staticmethod
    def filter_target_rerank_trec(
        save_dir,
        query_list,
        corpus_list,
        trec_file="dpr.trec",
        filtered_trec_file="dpr_filtered.trec",
    ):
        query_dict = {query["id"]: query for query in query_list}
        corpus_dict = {corpus["id"]: corpus for corpus in corpus_list}

        with open(f"{save_dir}/{trec_file}", "r") as f, open(
            f"{save_dir}/{filtered_trec_file}", "w"
        ) as f2:
            for line in f:
                qid, _, docid, rank, _, _ = line.strip().split(" ")

                query = query_dict[qid]
                corpus = corpus_dict[docid]

                query_entities = query["metadata"]["filter_entities"]
                text = corpus["text"]

                query_target_category = query["metadata"]["query_target_category"]

                if (
                    any([entity in text for entity in query_entities])
                    or query_target_category == "none"
                ):
                    f2.write(line)

    def benchmark_every_operator(
        self,
        save_dir,
        measures=["nDCG@20", "nDCG@100", "RR@100", "R@100"],
        execlude_non_numeric=False,
        execlude_only_numeric=False,
        skip_none=True,
    ):
        logging.info("=== Start benchmarking every operator ===")

        for corpus_encoder, _, name in self.encoders:
            logging.info(f"=== Start making corpus embedding for {name} ===")
            corpus_encoder_name = corpus_encoder.__name__

            corpus_emb_filename = f"corpus_emb_{name}_{corpus_encoder_name}.pkl"
            self.make_corpus_embedding(
                self.corpus_list, corpus_encoder, save_dir, corpus_emb_filename
            )

        # save corpus data
        with open(f"{save_dir}/corpus_list.json", "w") as f:
            json.dump(self.corpus_list, f, indent=4)

        operator_targets = ["equal", "less", "more", "around", "between"]

        if not skip_none:
            operator_targets.append("none")

        benchmark_results = {}
        for operator_target in operator_targets:
            operator_benchmark_data = self.benchmark_specific_operator(
                save_dir,
                operator_target,
                measures=measures,
                corpus_encode=False,
                execlude_non_numeric=execlude_non_numeric,
                execlude_only_numeric=execlude_only_numeric,
            )

            benchmark_results[operator_target] = operator_benchmark_data

        with open(f"{save_dir}/benchmark_every_operator.json", "w") as f:
            json.dump(benchmark_results, f, indent=4)

        return benchmark_results

    def benchmark_specific_operator(
        self,
        save_dir,
        operator_target,
        measures=["nDCG@20", "nDCG@100", "RR@100", "R@100"],
        corpus_encode=True,
        execlude_non_numeric=False,
        execlude_only_numeric=False,
    ):
        logging.info(f"=== Start benchmarking {operator_target} ===")

        operator_save_dir = f"{operator_target}"

        if not os.path.exists(f"{save_dir}/{operator_save_dir}"):
            os.makedirs(f"{save_dir}/{operator_save_dir}")

        query_list = self.generator.generate_query_list(
            query_operator_category=operator_target,
            query_target_category=self.query_target_category,
        )
        # save query
        with open(
            f"{save_dir}/{operator_save_dir}/{operator_target}_query_list.json", "w"
        ) as f:
            json.dump(query_list, f, indent=4)

        if not execlude_non_numeric:
            non_numeric_query_list = self.generator.generate_non_numeric_query_list(
                query_list
            )
            # save non numeric query
            with open(
                f"{save_dir}/{operator_save_dir}/{operator_target}_non_numeric_query_list.json",
                "w",
            ) as f:
                json.dump(non_numeric_query_list, f, indent=4)
        else:
            non_numeric_query_list = []

        if not execlude_only_numeric:
            if operator_target == "none":
                only_numeric_query_list = query_list
            else:
                only_numeric_query_list = (
                    self.generator.generate_only_numeric_query_list(query_list)
                )

            # save only numeric query
            with open(
                f"{save_dir}/{operator_save_dir}/{operator_target}_only_numeric_query_list.json",
                "w",
            ) as f:
                json.dump(only_numeric_query_list, f, indent=4)

        qrels = self.generator.generate_qrels(query_list, self.corpus_list)

        qrels_filename = f"{operator_save_dir}/qrels_{operator_target}.tsv"
        self.export_qrels(qrels, save_dir, filename=qrels_filename)

        benchmark_data = {}

        filenames = []

        ####################
        #  Dense Retrieval #
        ####################

        logging.info(f"=== Start Dense Retrieval ({operator_target}) ===")
        for corpus_encoder, query_encoder, name in self.encoders:
            query_encoder_name = query_encoder.__name__
            corpus_encoder_name = corpus_encoder.__name__

            ### File names
            corpus_emb_filename = f"corpus_emb_{name}_{corpus_encoder_name}.pkl"

            # normal
            query_emb_filename = f"{operator_save_dir}/query_emb_{operator_target}_{name}_{query_encoder_name}.pkl"
            trec_filename = f"{operator_save_dir}/{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}.trec"
            measures_filename = f"{operator_save_dir}/measures_{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}.tsv"

            # filtered normal
            filtered_trec_filename = f"{operator_save_dir}/{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_filtered.trec"
            filtered_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_filtered.tsv"

            temp_filenames = [
                query_emb_filename,
                trec_filename,
                measures_filename,
                filtered_trec_filename,
                filtered_measures_filename,
            ]

            if not execlude_non_numeric:
                # non numeric
                non_numeric_query_emb_filename = f"{operator_save_dir}/query_emb_{operator_target}_{name}_{query_encoder_name}_non_numeric.pkl"
                non_numeric_trec_filename = f"{operator_save_dir}/{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_non_numeric.trec"
                non_numeric_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_non_numeric.tsv"

                # filtered non numeric
                filtered_non_numeric_trec_filename = f"{operator_save_dir}/{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_non_numeric_filtered.trec"
                filtered_non_numeric_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_non_numeric_filtered.tsv"

                temp_filenames.extend(
                    [
                        non_numeric_query_emb_filename,
                        non_numeric_trec_filename,
                        non_numeric_measures_filename,
                        filtered_non_numeric_trec_filename,
                        filtered_non_numeric_measures_filename,
                    ]
                )

            if not execlude_only_numeric:
                # only numeric
                only_numeric_query_emb_filename = f"{operator_save_dir}/query_emb_{operator_target}_{name}_{query_encoder_name}_only_numeric.pkl"
                only_numeric_trec_filename = f"{operator_save_dir}/{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_only_numeric.trec"
                only_numeric_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_only_numeric.tsv"

                # filtered only numeric
                filtered_only_numeric_trec_filename = f"{operator_save_dir}/{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_only_numeric_filtered.trec"
                filtered_only_numeric_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{name}_{query_encoder_name}_{corpus_encoder_name}_only_numeric_filtered.tsv"

                temp_filenames.extend(
                    [
                        only_numeric_query_emb_filename,
                        only_numeric_trec_filename,
                        only_numeric_measures_filename,
                        filtered_only_numeric_trec_filename,
                        filtered_only_numeric_measures_filename,
                    ]
                )

            for temp_filename in temp_filenames:
                if temp_filename in filenames:
                    raise ValueError(f"Duplicate filename: {temp_filename}")

                filenames.append(temp_filename)

            logging.info(f"=== Start benchmarking {operator_target} with {name} ===")

            if corpus_encode:
                self.make_corpus_embedding(
                    self.corpus_list, corpus_encoder, save_dir, corpus_emb_filename
                )

            # normal
            self.make_query_embedding(
                query_list, query_encoder, save_dir, query_emb_filename
            )

            self.faiss_retriever_with_subprocess(
                save_dir,
                query_emb=query_emb_filename,
                corpus_emb=corpus_emb_filename,
                result_filename=trec_filename,
                depth=self.depth,
            )

            base_result = self.measure_retrieval_score_with_subprocess(
                save_dir,
                qrels_file=qrels_filename,
                trec_file=trec_filename,
                result_file=measures_filename,
                measures=measures,
            )

            # filtered normal
            self.filter_numeric_rerank_trec(
                save_dir,
                query_list,
                self.corpus_list,
                trec_file=trec_filename,
                filtered_trec_file=filtered_trec_filename,
            )

            filtered_base_result = self.measure_retrieval_score_with_subprocess(
                save_dir,
                qrels_file=qrels_filename,
                trec_file=filtered_trec_filename,
                result_file=filtered_measures_filename,
                measures=measures,
            )

            if not execlude_non_numeric:
                # non numeric
                self.make_query_embedding(
                    non_numeric_query_list,
                    query_encoder,
                    save_dir,
                    non_numeric_query_emb_filename,
                )

                self.faiss_retriever_with_subprocess(
                    save_dir,
                    query_emb=non_numeric_query_emb_filename,
                    corpus_emb=corpus_emb_filename,
                    result_filename=non_numeric_trec_filename,
                    depth=self.depth,
                )

                non_numerical_result = self.measure_retrieval_score_with_subprocess(
                    save_dir,
                    qrels_file=qrels_filename,
                    trec_file=non_numeric_trec_filename,
                    result_file=non_numeric_measures_filename,
                    measures=measures,
                )

                # filtered non numeric
                self.filter_numeric_rerank_trec(
                    save_dir,
                    query_list,
                    self.corpus_list,
                    trec_file=non_numeric_trec_filename,
                    filtered_trec_file=filtered_non_numeric_trec_filename,
                )

                filtered_non_numerical_result = (
                    self.measure_retrieval_score_with_subprocess(
                        save_dir,
                        qrels_file=qrels_filename,
                        trec_file=filtered_non_numeric_trec_filename,
                        result_file=filtered_non_numeric_measures_filename,
                        measures=measures,
                    )
                )

            if not execlude_only_numeric:
                # only numeric
                self.make_query_embedding(
                    only_numeric_query_list,
                    query_encoder,
                    save_dir,
                    only_numeric_query_emb_filename,
                )

                self.faiss_retriever_with_subprocess(
                    save_dir,
                    query_emb=only_numeric_query_emb_filename,
                    corpus_emb=corpus_emb_filename,
                    result_filename=only_numeric_trec_filename,
                    depth=self.depth,
                )

                only_numeric_result = self.measure_retrieval_score_with_subprocess(
                    save_dir,
                    qrels_file=qrels_filename,
                    trec_file=only_numeric_trec_filename,
                    result_file=only_numeric_measures_filename,
                    measures=measures,
                )

                # filtered only numeric
                self.filter_target_rerank_trec(
                    save_dir,
                    query_list,
                    self.corpus_list,
                    trec_file=only_numeric_trec_filename,
                    filtered_trec_file=filtered_only_numeric_trec_filename,
                )

                filtered_only_numeric_result = (
                    self.measure_retrieval_score_with_subprocess(
                        save_dir,
                        qrels_file=qrels_filename,
                        trec_file=filtered_only_numeric_trec_filename,
                        result_file=filtered_only_numeric_measures_filename,
                        measures=measures,
                    )
                )

            # write to benchmark_data
            measure_data = {
                "operator_target": operator_target,
                "result": self.format_measure(base_result),
                "filtered_result": self.format_measure(filtered_base_result),
            }

            if not execlude_non_numeric:
                measure_data["non_numeric_result"] = self.format_measure(
                    non_numerical_result
                )
                measure_data["filtered_non_numeric_result"] = self.format_measure(
                    filtered_non_numerical_result
                )

            if not execlude_only_numeric:
                measure_data["only_numeric_result"] = self.format_measure(
                    only_numeric_result
                )
                measure_data["filtered_only_numeric_result"] = self.format_measure(
                    filtered_only_numeric_result
                )

            benchmark_data[name] = measure_data

        ####################
        # Sparse Retrieval #
        ####################
        logging.info(f"=== Start Other Retrieval ({operator_target}) ===")
        for other_retriever in self.other_retrievers:
            other_retriever_name = other_retriever.__name__
            logging.info(f"=== Start benchmarking {other_retriever_name} ===")

            # normal
            sparse_trec_filename = (
                f"{operator_save_dir}/{operator_target}_{other_retriever_name}.trec"
            )
            sparse_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{other_retriever_name}.tsv"

            query_texts = list(map(lambda x: x["text"], query_list))
            query_ids = list(map(lambda x: x["id"], query_list))

            sparse_result = other_retriever.batch_search(
                query_texts, query_ids, top_k=self.depth
            )
            other_retriever.save_as_trec(
                sparse_result, f"{save_dir}/{sparse_trec_filename}"
            )

            sparse_measure = self.measure_retrieval_score_with_subprocess(
                save_dir,
                qrels_file=qrels_filename,
                trec_file=sparse_trec_filename,
                result_file=sparse_measures_filename,
                measures=measures,
            )

            # filtered
            filtered_sparse_trec_filename = f"{operator_save_dir}/{operator_target}_{other_retriever_name}_filtered.trec"
            filtered_sparse_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{other_retriever_name}_filtered.tsv"

            self.filter_numeric_rerank_trec(
                save_dir,
                query_list,
                self.corpus_list,
                trec_file=sparse_trec_filename,
                filtered_trec_file=filtered_sparse_trec_filename,
            )

            filtered_sparse_measure = self.measure_retrieval_score_with_subprocess(
                save_dir,
                qrels_file=qrels_filename,
                trec_file=filtered_sparse_trec_filename,
                result_file=filtered_sparse_measures_filename,
                measures=measures,
            )

            if not execlude_non_numeric:
                non_numeric_query_texts = list(
                    map(lambda x: x["text"], non_numeric_query_list)
                )

                non_numeric_sparse_trec_filename = f"{operator_save_dir}/{operator_target}_{other_retriever_name}_non_numeric.trec"
                non_numeric_sparse_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{other_retriever_name}_non_numeric.tsv"

                non_numeric_sparse_result = other_retriever.batch_search(
                    non_numeric_query_texts, query_ids, top_k=self.depth
                )
                other_retriever.save_as_trec(
                    non_numeric_sparse_result,
                    f"{save_dir}/{non_numeric_sparse_trec_filename}",
                )

                non_numeric_sparse_measure = (
                    self.measure_retrieval_score_with_subprocess(
                        save_dir,
                        qrels_file=qrels_filename,
                        trec_file=non_numeric_sparse_trec_filename,
                        result_file=non_numeric_sparse_measures_filename,
                        measures=measures,
                    )
                )

                # filtered
                non_numeric_filtered_sparse_trec_filename = f"{operator_save_dir}/{operator_target}_{other_retriever_name}_non_numeric_filtered.trec"
                non_numeric_filtered_sparse_measures_filename = f"{operator_save_dir}/measures_{operator_target}_{other_retriever_name}_non_numeric_filtered.tsv"

                self.filter_numeric_rerank_trec(
                    save_dir,
                    query_list,
                    self.corpus_list,
                    trec_file=sparse_trec_filename,
                    filtered_trec_file=non_numeric_filtered_sparse_trec_filename,
                )

                non_numeric_filtered_sparse_measure = (
                    self.measure_retrieval_score_with_subprocess(
                        save_dir,
                        qrels_file=qrels_filename,
                        trec_file=filtered_sparse_trec_filename,
                        result_file=non_numeric_filtered_sparse_measures_filename,
                        measures=measures,
                    )
                )

            # write to benchmark_data
            measure_data = {
                "operator_target": operator_target,
                "result": self.format_measure(sparse_measure),
                "filtered_result": self.format_measure(filtered_sparse_measure),
            }

            if not execlude_non_numeric:
                measure_data["non_numeric_result"] = self.format_measure(
                    non_numeric_sparse_measure
                )
                measure_data["filtered_non_numeric_result"] = self.format_measure(
                    non_numeric_filtered_sparse_measure
                )

            benchmark_data[other_retriever_name] = measure_data

        with open(f"{save_dir}/benchmark_{operator_target}.json", "w") as f:
            json.dump(benchmark_data, f, indent=4)

        return benchmark_data
