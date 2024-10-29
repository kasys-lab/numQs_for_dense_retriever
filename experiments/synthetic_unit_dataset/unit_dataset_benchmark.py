import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import numpy as np

from experimental_settings.numerical_condition_benchmark_base import (
    NumericalConditionDatasetBuilderBase,
)


class ArtificialUnitDataset(NumericalConditionDatasetBuilderBase):
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
        query_pattern_filepath=None,
        query_num_range=[1, 100],
        query_num_step=1,
    ):
        self.query_num_range = query_num_range
        self.query_num_step = query_num_step

        np.random.seed(seed)

        self.query_pattern_filepath = query_pattern_filepath
        self.query_patterns = self._load_query_patterns(query_pattern_filepath)
        super().__init__(
            csv_data_filepath,
            entity_category_primary_column,
            entity_category_secondary_column,
            number_column,
            max_corpus,
            query_num,
            only_save_relevance,
            is_primary_category_list,
            is_secondary_category_list,
            seed=seed,
            query_nums_pick_replace=query_nums_pick_replace,
        )

    def pickup_query_numbers(self, query_num):
        query_numbers = [
            i
            for i in range(
                self.query_num_range[0], self.query_num_range[1], self.query_num_step
            )
        ]

        self.query_num = len(query_numbers)
        self.query_numbers = query_numbers
        self.query_numbers_secondary = np.random.choice(
            query_numbers, self.query_num, replace=False
        ).tolist()

    @staticmethod
    def _load_query_patterns(query_pattern_filepath):
        with open(query_pattern_filepath, "r") as f:
            query_patterns = json.load(f)

        return query_patterns

    def generate_corpus_text(  # type: ignore
        self, primary_category: str, secondary_category: str, number, metadata
    ):
        text = metadata["text"]

        corpus_text = text

        return corpus_text

    def generate_none_query_target(self):
        return ""

    def generate_equal_query_text(self, query_target, query_number):
        query_text_pattern = self.query_patterns["equal"]

        query_text = query_text_pattern.format(number=query_number)

        return query_text

    def generate_less_query_text(self, query_target, query_number):
        query_text_pattern = self.query_patterns["less"]

        query_text = query_text_pattern.format(number=query_number)

        return query_text

    def generate_more_query_text(self, query_target, query_number):
        query_text_pattern = self.query_patterns["more"]

        query_text = query_text_pattern.format(number=query_number)

        return query_text

    def generate_between_query_text(self, query_target, query_number):
        lower_bound, upper_bound = sorted(query_number)

        query_text_pattern = self.query_patterns["between"]

        query_text = query_text_pattern.format(
            number_min=lower_bound, number_max=upper_bound
        )

        return query_text

    def generate_around_query_text(self, query_target, query_number):
        query_text_pattern = self.query_patterns["around"]

        query_text = query_text_pattern.format(number=query_number)

        return query_text

    def generate_none_query_text(self, query_target):
        query_text = self.query_patterns["none"]

        return query_text
